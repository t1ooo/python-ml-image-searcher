from contextlib import contextmanager
import hashlib
import itertools
from PIL import Image as PImage, UnidentifiedImageError
from sentence_transformers import SentenceTransformer
from torch.nn.functional import cosine_similarity
from transformers import CLIPProcessor, CLIPModel, pipeline
from typing import Callable, Generator, Iterable, Protocol, TypeVar
import logging
import numpy
import os
import pickle
import torch

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def _scan_dirs(dirs: Iterable[str]) -> Generator[str, None, None]:
    for dir in dirs:
        for entry in os.scandir(dir):
            yield entry.path


def _extension_filter(exts: list[str]) -> Callable[[str], bool]:
    def f(path: str) -> bool:
        return path.split(".")[-1] in exts

    return f


def _makedirs(dir: str):
    if dir != "":
        os.makedirs(dir, exist_ok=True)


def scan_image_dirs(
    dirs: Iterable[str], limit: int, exts: list[str] = ["png", "jpg", "jpeg", "gif"]
) -> Iterable[str]:
    """Return an iterator of files found in directories, filtered by extension and limited by limit."""
    paths = _scan_dirs(dirs)
    paths = filter(_extension_filter(exts), paths)
    paths = itertools.islice(paths, limit)
    return paths


def _norm(tensor: torch.Tensor) -> torch.Tensor:
    return tensor / tensor.norm(dim=-1, keepdim=True)


def _md5(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()


def _signature(*args: str) -> str:
    return _md5(("-").join(args))


class Model(Protocol):
    def get_image_embeddings(self, imgs: list[PImage.Image]) -> torch.Tensor:
        """Return a tensor with calculated image embeddings."""
        ...

    def get_text_embeddings(self, texts: list[str]) -> torch.Tensor:
        """Return a tensor with calculated text embeddings."""
        ...

    @property
    def signature(self) -> str:
        """Return the model signature."""
        ...


# TODO: try feature-extraction pipeline(https://huggingface.co/tasks/feature-extraction) instead of SentenceTransformer for faster indexing
class ImageToTextModel(Model):
    def __init__(
        self,
        img_to_txt_model_name: str = "ydshieh/vit-gpt2-coco-en",
        sent_transformer_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        max_new_tokens: int = 20,
    ):
        self._i2t_pipeline = pipeline("image-to-text", model=img_to_txt_model_name)
        self._st_model: SentenceTransformer = SentenceTransformer(
            sent_transformer_model_name
        )
        self._max_new_tokens = max_new_tokens

        self._signature = _signature(
            self.__class__.__name__,
            img_to_txt_model_name,
            sent_transformer_model_name,
            str(max_new_tokens),
        )

    def get_image_embeddings(self, imgs: list[PImage.Image]) -> torch.Tensor:
        descriptions = [
            v[0]["generated_text"]  # type: ignore
            for v in self._i2t_pipeline(imgs, max_new_tokens=self._max_new_tokens)  # type: ignore
        ]
        return self.get_text_embeddings(descriptions)  # type: ignore

    def get_text_embeddings(self, texts: list[str]) -> torch.Tensor:
        return self._st_model.encode(texts, convert_to_tensor=True)  # type: ignore

    @property
    def signature(self) -> str:
        return self._signature


class ClipModel(Model):
    def __init__(
        self, model_name: str = "openai/clip-vit-base-patch16", norm: bool = False
    ):
        self._model = CLIPModel.from_pretrained(model_name)
        self._processor = CLIPProcessor.from_pretrained(model_name)
        self._norm = norm

        self._signature = _signature(
            self.__class__.__name__,
            model_name,
            str(norm),
        )

    def get_image_embeddings(self, imgs: list[PImage.Image]) -> torch.Tensor:
        p = self._processor(images=imgs, return_tensors="pt", padding=True)
        r = self._model.get_image_features(**p)  # type: ignore
        if self._norm:
            r = _norm(r)
        return r

    def get_text_embeddings(self, texts: list[str]) -> torch.Tensor:
        p = self._processor(text=texts, return_tensors="pt", padding=True)
        r = self._model.get_text_features(**p)  # type: ignore
        if self._norm:
            r = _norm(r)
        return r

    @property
    def signature(self) -> str:
        return self._signature


class AppException(Exception):
    pass


class ImagePreprocessor(Protocol):
    def process(self, pimg: PImage.Image) -> PImage.Image:
        ...


class Resizer(ImagePreprocessor):
    def __init__(self, max_size: int = 224):
        self._max_size = max_size

    def process(self, pimg: PImage.Image) -> PImage.Image:
        width, height = pimg.size
        ratio = self._max_size / min([width, height])
        width, height = int(width * ratio), int(height * ratio)
        return pimg.resize((width, height))


_T = TypeVar("_T")


def _batched(iterable: Iterable[_T], size: int) -> Generator[Iterable[_T], None, None]:
    iterator = iter(iterable)
    while chunk := tuple(itertools.islice(iterator, size)):
        yield chunk


Item = tuple[list[str], torch.Tensor]


@contextmanager
def _pickle_writer(file: str, mode: str = "wb"):
    _makedirs(os.path.dirname(file))
    f = open(file, mode)
    try:
        def fn(item: Item):
            pickle.dump(item, f)

        yield fn
    finally:
        f.close()


def _pickle_reader(file: str, mode: str = "rb") -> Generator[Item, None, None]:
    with open(file, mode) as f:
        while True:
            try:
                paths, embs = pickle.load(f)
                yield (paths, embs)
            except EOFError:
                break


def _load_pimages(
    in_paths: Iterable[str], img_preprocessor: ImagePreprocessor | None = None
) -> tuple[list[str], list[PImage.Image]]:
    paths: list[str] = []
    pimgs: list[PImage.Image] = []
    for path in in_paths:
        try:
            pimg = PImage.open(path)
            if img_preprocessor:
                pimg = img_preprocessor.process(pimg)

            paths.append(path)
            pimgs.append(pimg)
        except UnidentifiedImageError:
            logger.warning(f"Failed to read file as image {path}")
            pass
    return paths, pimgs


# TODO: try a vector database for embeddings like milvus(https://milvus.io/)
class App:
    def __init__(
        self,
        model: Model,
        index_dir: str, # directory to save the index
        img_preprocessor: ImagePreprocessor | None = None,
        batch_size: int = 25, # reduce batch size if you get out of memory
    ):
        self._index_fname = f"{index_dir}/index-{model.signature}.pickle"
        self._model = model
        self._img_preprocessor = img_preprocessor
        self._batch_size = batch_size
        self._paths: list[str] | None = None
        self._embs: torch.Tensor | None = None

    def build_index(self, img_paths: Iterable[str]):
        """Build an index for the given image paths."""
        total = 0
        with _pickle_writer(self._index_fname) as writer:
            for b_img_paths in _batched(img_paths, self._batch_size):
                try:
                    b_paths, b_pimgs = _load_pimages(
                        b_img_paths, self._img_preprocessor
                    )
                    b_embs = self._model.get_image_embeddings(b_pimgs)
                    assert len(b_paths) == len(b_embs)
                    writer((b_paths, b_embs))
                    total += len(b_paths)
                    logger.info(f"Build index {total}...")
                except KeyboardInterrupt:
                    logger.info(f"Build index stopped by user.")

    def search(self, query: str, limit: int = 10) -> list[tuple[str, float]]:
        """Return a list of tuples (image path, cosine similarity) matching the given query, sorted in descending order by cosine similarity."""
        try:
            reader = _pickle_reader(self._index_fname)
        except pickle.PickleError as e:
            logger.error(e)
            raise AppException(
                "Failed to load index. Maybe you forgot to build an index?"
            )

        query_embed = self._model.get_text_embeddings([query])

        paths = []
        cos_sim = []
        for b_paths, b_embs in reader:
            b_cos_sim = cosine_similarity(b_embs, query_embed).tolist()
            paths.extend(b_paths)
            cos_sim.extend(b_cos_sim)

        assert len(paths) == len(cos_sim)
        logger.info(f"Search through {len(paths)} images.")

        idxs = numpy.argsort(cos_sim)[::-1][:limit]
        return [(paths[i], cos_sim[i]) for i in idxs]

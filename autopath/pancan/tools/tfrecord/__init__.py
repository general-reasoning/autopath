from functools import partial
import io
import os
import struct
from typing import Optional, Tuple, Iterable, Union, List, Dict, Callable, Any


import gzip
import numpy as np

import torch
import torch.utils.data
import torchvision.io

import imghdr

import dbx

from . import protobuf as example_pb2


TYPENAME_MAPPING = {
    "byte": "bytes_list",
    "float": "float_list",
    "int": "int64_list"
}

FEATURE_DESCRIPTION = {
    'image_raw': 'byte',
    'slide': 'byte',
    'loc_x': 'int',
    'loc_y': 'int'
}


SUPPORTED_FORMATS = ['svs', 'tif', 'ndpi', 'vms', 'vmu', 'scn', 'mrxs',
                     'tiff', 'svslide', 'bif', 'jpg', 'jpeg', 'png',
                     'ome.tif', 'ome.tiff']


def get_tfrecord_by_index(
    tfrecord: str,
    index: int,
    *,
    compression_type: Optional[str] = None,
    index_array: Optional[np.ndarray] = None
) -> Dict:
    """Read a specific record in a TFRecord file.

    Args:
        tfrecord (str): TFRecord file to read.
        index (int): Index of record to read from the file.
        compression_type (str): Type of compression in the TFRecord file.
            Either 'gzip' or None. Defaults to None.

    Returns:
        A dictionary mapping record names (e.g., ``'slide'``, ``'image_raw'``,
        ``'loc_x'``, and ``'loc_y'``) to their values. ``'slide'`` will be a
        string, ``image_raw`` will be bytes, and ``'loc_x'`` and ``'loc_y'``
        will be `int`.

    """

    # Load the TFRecord file.
    if compression_type == "gzip":
        file = gzip.open(tfrecord, 'rb')
    elif compression_type is None:
        file = io.open(tfrecord, 'rb')  # type: ignore
    else:
        raise ValueError("compression_type should be 'gzip' or None")
    if not os.path.getsize(tfrecord):
        raise ValueError(f"{tfrecord} is empty.")

    # Load the TFRecord index file.
    if index:
        idx = index_array if index_array is not None else load_index(tfrecord)
        if idx is None:
            raise ValueError(f"Could not find tfrecord index for {tfrecord}")
        if index >= idx.shape[0]:
            raise ValueError(
                f"Index {index} is invalid for tfrecord {tfrecord} "
                f"(size: {idx.shape[0]})"
            )
        start_offset = idx[index, 0]
        file.seek(start_offset)

    # Read the designated record.
    length_bytes = bytearray(8)
    crc_bytes = bytearray(4)
    datum_bytes = bytearray(1024 * 1024)
    if file.readinto(length_bytes) != 8:
        raise RuntimeError("Failed to read the record size.")
    if file.readinto(crc_bytes) != 4:
        raise RuntimeError("Failed to read the start token.")
    length, = struct.unpack("<Q", length_bytes)
    if length > len(datum_bytes):
        try:
            _fill = int(length * 1.5)
            datum_bytes = datum_bytes.zfill(_fill)
        except OverflowError:
            raise OverflowError('Error reading tfrecords; please '
                                'try regenerating index files')
    datum_bytes_view = memoryview(datum_bytes)[:length]
    if file.readinto(datum_bytes_view) != length:
        raise RuntimeError("Failed to read the record.")
    if file.readinto(crc_bytes) != 4:
        raise RuntimeError("Failed to read the end token.")

    # Process record bytes.
    try:
        record = process_record_from_bytes(datum_bytes_view)
    except KeyError:
        raise ValueError(
            f'Unable to detect TFRecord format: {tfrecord}'
        )

    file.close()
    return record


def path_to_name(path: str) -> str:
    '''Returns name of a file, without extension,
    from a given full path string.'''
    _file = os.path.basename(path)
    dot_split = _file.split('.')
    if len(dot_split) == 1:
        return _file
    elif len(dot_split) > 2 and '.'.join(dot_split[-2:]) in SUPPORTED_FORMATS:
        return '.'.join(dot_split[:-2])
    else:
        return '.'.join(dot_split[:-1])
    

def find_index(tfrecord: str) -> Optional[str]:
    """Find the index file for a TFRecord."""
    from os.path import dirname, join, exists
    name = path_to_name(tfrecord)
    if exists(join(dirname(tfrecord), name+'.index')):
        return join(dirname(tfrecord), name+'.index')
    elif exists(join(dirname(tfrecord), name+'.index.npz')):
        return join(dirname(tfrecord), name+'.index.npz')
    elif exists(join(dirname(tfrecord), name+'.index.npy')):
        return join(dirname(tfrecord), name+'.index.npy')
    else:
        return None
    

def load_index(tfrecord: str) -> Optional[np.ndarray]:
    """Find and load the index associated with a TFRecord."""
    index_path = find_index(tfrecord)
    if index_path is None:
        raise OSError(f"Could not find index path for TFRecord {tfrecord}")
    if os.stat(index_path).st_size == 0:
        return None
    elif index_path.endswith('npz'):
        return np.load(index_path)['arr_0']
    elif index_path.endswith('npy'):
        return np.load(index_path)
    else:
        return np.loadtxt(index_path, dtype=np.int64)
    

def process_record_from_bytes(bytes_view):
    try:
        record = process_record(bytes_view)
    except KeyError:
        feature_description = {
            k: v for k, v in FEATURE_DESCRIPTION.items()
            if k in ('slide', 'image_raw')
        }
        record = process_record(bytes_view, description=feature_description)

    # Final parsing.
    if 'slide' in record:
        record['slide'] = bytes(record['slide']).decode('utf-8')
    if 'image_raw' in record:
        record['image_raw'] = bytes(record['image_raw'])
    if 'loc_x' in record:
        record['loc_x'] = record['loc_x'][0]
    if 'loc_y' in record:
        record['loc_y'] = record['loc_y'][0]
    return record


def process_feature(
    feature: example_pb2.Feature,  # type: ignore
    typename: str,
    typename_mapping: Dict,
    key: str
) -> np.ndarray:
    # NOTE: We assume that each key in the example has only one field
    # (either "bytes_list", "float_list", or "int64_list")!
    field = feature.ListFields()[0]  # type: ignore
    inferred_typename, value = field[0].name, field[1].value

    if typename is not None:
        tf_typename = typename_mapping[typename]
        if tf_typename != inferred_typename:
            reversed_mapping = {v: k for k, v in typename_mapping.items()}
            raise TypeError(
                f"Incompatible type '{typename}' for `{key}` "
                f"(should be '{reversed_mapping[inferred_typename]}')."
            )

    if inferred_typename == "bytes_list":
        value = np.frombuffer(value[0], dtype=np.uint8)
    elif inferred_typename == "float_list":
        value = np.array(value, dtype=np.float32)
    elif inferred_typename == "int64_list":
        value = np.array(value, dtype=np.int64)
    return value


def extract_feature_dict(
    features: Union[example_pb2.FeatureLists,  # type: ignore
                    example_pb2.Features],  # type: ignore
    description: Optional[Union[List, Dict]],
    typename_mapping: Dict
) -> Dict[str, Any]:
    if isinstance(features, example_pb2.FeatureLists):
        features = features.feature_list  # type: ignore

        def get_value(typename, typename_mapping, key):
            feature = features[key].feature
            fn = partial(
                process_feature,
                typename=typename,
                typename_mapping=typename_mapping,
                key=key
            )
            return list(map(fn, feature))
    elif isinstance(features, example_pb2.Features):
        features = features.feature  # type: ignore

        def get_value(typename, typename_mapping, key):
            return process_feature(features[key], typename,
                                   typename_mapping, key)
    else:
        raise TypeError(f"Incompatible type: features should be either of type "
                        f"example_pb2.Features or example_pb2.FeatureLists and "
                        f"not {type(features)}")

    all_keys = list(features.keys())  # type: ignore

    if description is None or len(description) == 0:
        description = dict.fromkeys(all_keys, None)
    elif isinstance(description, list):
        description = dict.fromkeys(description, None)

    processed_features = {}
    for key, typename in description.items():
        if key not in all_keys:
            raise KeyError(f"Key {key} doesn't exist (select from {all_keys})!")

        processed_features[key] = get_value(typename, typename_mapping, key)

    return processed_features


def process_record(record, description=None):
    if description is None:
        description = FEATURE_DESCRIPTION
    example = example_pb2.Example()
    example.ParseFromString(record)
    return extract_feature_dict(
        example.features,
        description,
        TYPENAME_MAPPING)


def cwh_to_whc(img: torch.Tensor) -> torch.Tensor:
    """Convert torch tensor from C x W x H => W x H x C"""
    if len(img.shape) == 3:
        return img.permute(1, 2, 0)  # CWH -> WHC
    elif len(img.shape) == 4:
        return img.permute(0, 2, 3, 1)  # BCWH -> BWHC
    else:
        raise ValueError(
            "Invalid shape for channel conversion. Expected 3 or 4 dims, "
            f"got {len(img.shape)} (shape={img.shape})")


def detect_tfrecord_format(tfr: str) -> Tuple[Optional[List[str]],
                                              Optional[str]]:
    '''Detects tfrecord format.

    Args:
        tfr (str): Path to tfrecord.

    Returns:
        A tuple containing

            list(str): List of detected features.

            str: Image file type (png/jpeg)
    '''
    try:
        record = get_tfrecord_by_index(tfr, index=0)
    except ValueError:
        dbx.Logger().error(f"Unable to detect format for {tfr}; file empty.")
        return None, None
    img_type = imghdr.what('', record['image_raw'])
    return list(record.keys()), img_type


def decode_image(
    image: Union[bytes, str, torch.Tensor],
    *,
    img_type: Optional[str] = None,
    device: Optional[torch.device] = None,
    transform: Optional[Any] = None,
) -> torch.Tensor:
    """Decodes image string/bytes to Tensor (W x H x C).

    Args:
        image (Union[bytes, str, torch.Tensor]): Image to decode.

    Keyword args:
        img_type (str, optional): Image type. Defaults to None.
        device (torch.device, optional): Device to move image to.
            Defaults to None.
        transform (Callable, optional): Arbitrary torchvision transform function.
            Performs transformation after augmentations but before standardization.
            Defaults to None.

    """
    if img_type != 'numpy':
        #np_data = torch.from_numpy(np.fromstring(image, dtype=np.uint8))
        np_data = torch.from_numpy(np.frombuffer(image, dtype=np.uint8).copy())
        image = cwh_to_whc(torchvision.io.decode_image(np_data))
        # Alternative method using PIL decoding:
        # image = np.array(Image.open(BytesIO(img_string)))
        # Remove alpha, if present.
        if image.shape[-1] == 4:
            image = image[..., :3]

    assert isinstance(image, torch.Tensor)

    if device is not None:
        image = image.to(device)

    if transform is not None:
        image = transform(image)

    return image


def get_tfrecord_parser(
    tfrecord_path: str,
    features_to_return: Iterable[str] = None,
    decode_images: bool = True,
    **kwargs,
) -> Callable:

    """Gets tfrecord parser using dareblopy reader.

    Args:
        tfrecord_path (str): Path to tfrecord to parse.
        features_to_return (list or dict, optional): Designates format for how
            features should be returned from parser. If a list of feature names
            is provided, the parsing function will return tfrecord features as
            a list in the order provided. If a dictionary of labels (keys)
            mapping to feature names (values) is provided, features will be
            returned from the parser as a dictionary matching the same format.
            If None, will return all features as a list.
        decode_images (bool, optional): Decode raw image strings into image
            arrays. Defaults to True.
        standardize (bool, optional): Standardize images into the range (0,1).
            Defaults to False.
        augment (str or bool): Image augmentations to perform. Augmentations include:

            * ``'x'``: Random horizontal flip
            * ``'y'``: Random vertical flip
            * ``'r'``: Random 90-degree rotation
            * ``'j'``: Random JPEG compression (50% chance to compress with quality between 50-100)
            * ``'b'``: Random Gaussian blur (10% chance to blur with sigma between 0.5-2.0)

            Combine letters to define augmentations, such as ``'xyrjn'``.
            A value of True will use ``'xyrjb'``.
            Note: this function does not support stain augmentation.

    Returns:
        A tuple containing

            func: Parsing function

            dict: Detected feature description for the tfrecord
    """

    features, img_type = detect_tfrecord_format(tfrecord_path)
    if features is None or img_type is None:
        raise ValueError(f"Unable to read TFRecord {tfrecord_path}")
    if features_to_return is None:
        features_to_return = {k: k for k in features}
    elif not all(f in features for f in features_to_return):
        detected = ",".join(features)
        _ftrs = list(features_to_return.keys())  # type: ignore
        raise ValueError(
            f'Not all features {",".join(_ftrs)} '
            f'were found in the tfrecord {detected}'
        )

    parser = TFRecordParser(
        features_to_return,
        decode_images,
        img_type,
    )
    return parser

# -------------------------------------------------------------------------

class TFRecordParser:

    def __init__(self, features_to_return, decode_images, img_type, transform=None):
        self.features_to_return = features_to_return
        self.decode_images = decode_images
        self.img_type = img_type
        self.transform = transform

    def __call__(self, record):
        """Each item in args is an array with one item, as the dareblopy reader
        returns items in batches and we have set our batch_size = 1 for
        interleaving.
        """
        features = {}
        if ('slide' in self.features_to_return):
            slide = bytes(record['slide']).decode('utf-8')
            features['slide'] = slide
        if ('image_raw' in self.features_to_return):
            img = bytes(record['image_raw'])
            if self.decode_images:
                features['image_raw'] = decode_image(
                    img,
                    img_type=self.img_type,
                    transform=self.transform
                )
            else:
                features['image_raw'] = img
        if ('loc_x' in self.features_to_return):
            features['loc_x'] = record['loc_x'][0]
        if ('loc_y' in self.features_to_return):
            features['loc_y'] = record['loc_y'][0]
        if type(self.features_to_return) == dict:
            return {
                label: features[f]
                for label, f in self.features_to_return.items()
            }
        else:
            return [features[f] for f in self.features_to_return]


class TFRecordIterator:
    typename_mapping = {
        "byte": "bytes_list",
        "float": "float_list",
        "int": "int64_list"
    }

    def __init__(
        self,
        data_path: str,
        index: Optional[np.ndarray] = None,
        shard: Optional[Tuple[int, int]] = None,
        clip: Optional[int] = None,
        compression_type: Optional[str] = None,
        random_start: bool = False,
        datum_bytes: Optional[bytearray] = None,
    ) -> None:
        """Create an iterator over the tfrecord dataset.

        Since the tfrecords file stores each example as bytes, we can
        define an iterator over `datum_bytes_view`, which is a memoryview
        object referencing the bytes.

        Params:
        -------
        data_path: str
            TFRecord file path.

        index: optional, default=None
            np.loadtxt(index_path, dtype=np.int64)

        shard: tuple of ints, optional, default=None
            A tuple (index, count) representing worker_id and num_workers
            count. Necessary to evenly split/shard the dataset among many
            workers (i.e. >1).

        random_start: randomize starting location of reading.
            Requires an index file. Only works if shard is None.

        Yields:
        -------
        datum_bytes_view: memoryview
            Object referencing the specified `datum_bytes` contained in the
            file (for a single record).
        """

        if compression_type == "gzip":
            self.file = gzip.open(data_path, 'rb')
        elif compression_type is None:
            self.file = io.open(data_path, 'rb')  # type: ignore
        else:
            raise ValueError("compression_type should be 'gzip' or None")

        self.data_path = data_path
        self.shard = shard
        self.clip = clip
        self.random_start = random_start
        if datum_bytes is not None:
            self.datum_bytes = datum_bytes
        else:
            self.datum_bytes = bytearray(1024 * 1024)
        self.length_bytes = bytearray(8)
        self.crc_bytes = bytearray(4)
        self.index = index
        self.index_is_nonsequential = None
        if self.index is not None and len(self.index) != 0:
            # For the case that there is only a single record in the file
            if len(self.index.shape) == 1:
                self.index = np.expand_dims(self.index, axis=0)

            # Check if the index file contains sequential records
            self.index_is_nonsequential = (
                not np.all(np.cumsum(self.index[:, 1][:-1])
                           + self.index[0, 0] == self.index[:, 0][1:])
            )

            # Only keep the starting bytes for the indices
            self.index = self.index[:, 0]  # type: ignore

            # Ensure the starting bytes are in order
            self.index = np.sort(self.index)

    def _read_sequential_records(self, start_offset=None, end_offset=None):
        """Read sequential records from the given starting byte."""
        if start_offset is not None:
            self.file.seek(start_offset)
        if end_offset is None:
            end_offset = os.path.getsize(self.data_path)
        while self.file.tell() < end_offset:
            yield self._read_next_data()

    def _read_nonsequential_records(self, start_offset=None, end_offset=None):
        """Read nonsequential records from the given starting byte.

        Only read records with starting bytes reflected in the index file.
        """
        if start_offset not in self.index:
            raise ValueError("Offset not in the tfrecord index.")
        if start_offset is None:
            start_offset = self.index[0]
            index_loc = 0
        else:
            index_loc = np.argwhere(self.index == start_offset)[0][0]

        if end_offset is None:
            end_offset = os.path.getsize(self.data_path)

        while self.index[index_loc] < end_offset:
            if self.file.tell() != self.index[index_loc]:
                self.file.seek(self.index[index_loc])

            yield self._read_next_data()
            index_loc += 1

            # End the loop if we have reached the last index
            if index_loc >= len(self.index):
                break

    def _read_next_data(self) -> memoryview:
        """Read the next record from the tfrecord file."""
        try:
            data = self._read_data(
                self.file,
                self.length_bytes,
                self.crc_bytes,
                self.datum_bytes
            )
        except Exception as e:
            dbx.Logger().error("Error reading data from tfrecord {}: {}".format(
                self.data_path, e
            ))
            raise e
        try:
            return self.process(data)
        except Exception as e:
            dbx.Logger().error("Error processing data from tfrecord {}: {}".format(
                self.data_path, e
            ))
            raise e

    def read_records(self, start_offset=None, end_offset=None):
        if self.index_is_nonsequential:
            yield from self._read_nonsequential_records(start_offset, end_offset)
        else:
            yield from self._read_sequential_records(start_offset, end_offset)

    def __iter__(self) -> Iterable[memoryview]:
        """Create the iterator."""

        if self.index is None:
            yield from self.read_records()
        elif not len(self.index):
            return
        else:
            if self.clip:
                if self.clip == len(self.index):
                    clip_offset = None
                else:
                    clip_offset = self.index[self.clip]
                self.index = self.index[:self.clip]
            else:
                clip_offset = None
            if self.shard is None and self.random_start:
                assert self.index is not None
                offset = np.random.choice(self.index)
                yield from self.read_records(offset, clip_offset)
                yield from self.read_records(0, offset)
            elif self.shard is None:
                yield from self.read_records(0, clip_offset)
            else:
                shard_idx, shard_count = self.shard
                all_shard_indices = np.array_split(self.index, shard_count)
                if shard_count >= self.index.shape[0]:  # type: ignore
                    # There are fewer records than shards, so
                    # only the first shard will read
                    if shard_idx == 0:
                        start_byte = all_shard_indices[shard_idx][0]
                        yield from self.read_records(start_byte, clip_offset)
                        return
                    else:
                        return
                elif shard_idx < (shard_count-1):
                    end_byte = all_shard_indices[shard_idx + 1][0]
                else:
                    end_byte = clip_offset
                start_byte = all_shard_indices[shard_idx][0]
                yield from self.read_records(start_byte, end_byte)

    def process(self, record):
        return record

    def close(self):
        self.file.close()

    def __del__(self):
        try:
            self.file.close()
        except Exception:
            pass

    @staticmethod
    def _read_data(file, length_bytes, crc_bytes, datum_bytes) -> memoryview:
        """Read the next record from the tfrecord file."""
        if file.readinto(length_bytes) != 8:
            raise RuntimeError("Failed to read the record size.")
        if file.readinto(crc_bytes) != 4:
            raise RuntimeError("Failed to read the start token.")
        length, = struct.unpack("<Q", length_bytes)
        if length > len(datum_bytes):
            try:
                _fill = int(length * 1.5)
                datum_bytes = datum_bytes.zfill(_fill)
            except OverflowError:
                raise OverflowError('Overflow encountered reading tfrecords; please '
                                    'try regenerating index files')
        datum_bytes_view = memoryview(datum_bytes)[:length]
        if file.readinto(datum_bytes_view) != length:
            raise RuntimeError("Failed to read the record.")
        if file.readinto(crc_bytes) != 4:
            raise RuntimeError("Failed to read the end token.")
        return datum_bytes_view
    

class ExampleIterator(TFRecordIterator):
    def __init__(
        self,
        data_path: str,
        index: Optional[np.ndarray] = None,
        shard: Optional[Tuple[int, int]] = None,
        clip: Optional[int] = None,
        compression_type: Optional[str] = None,
        random_start: bool = False,
        datum_bytes: Optional[bytearray] = None,
        description: Union[List[str], Dict[str, str], None] = None,
    ):
        """
        description: list or dict of str, optional, default=None
            List of keys or dict of (key, value) pairs to extract from each
            record. The keys represent the name of the features and the
            values ("byte", "float", or "int") correspond to the data type.
            If dtypes are provided, then they are verified against the
            inferred type for compatibility purposes. If None (default),
            then all features contained in the file are extracted.
        """
        super().__init__(
            data_path,
            index,
            shard,
            clip,
            compression_type,
            random_start,
            datum_bytes
        )
        self.description = description

    def process(self, record):
        example = example_pb2.Example()
        example.ParseFromString(record)
        return extract_feature_dict(
            example.features,
            self.description,
            self.typename_mapping
        )
    

def tfrecord_loader(
    data_path: str,
    index: Optional[np.ndarray] = None,
    description: Union[List[str], Dict[str, str], None] = None,
    shard: Optional[Tuple[int, int]] = None,
    clip: Optional[int] = None,
    compression_type: Optional[str] = None,
    datum_bytes: Optional[bytearray] = None,
) -> Iterable[Union[
        Dict[str, np.ndarray],
        Tuple[Dict[str, np.ndarray], Dict[str, List[np.ndarray]]]]]:
    """Create an iterator over the (decoded) examples contained within
    the dataset.

    Decodes raw bytes of the features (contained within the dataset)
    into its respective format.

    Params:
    -------
    data_path: str
        TFRecord file path.

    index_path: np.ndarray or None
        Loaded index. Can be set to None if no file is available.

    description: list or dict of str, optional, default=None
        List of keys or dict of (key, value) pairs to extract from each
        record. The keys represent the name of the features and the
        values ("byte", "float", or "int") correspond to the data type.
        If dtypes are provided, then they are verified against the
        inferred type for compatibility purposes. If None (default),
        or an empty list or dictionary, then all features contained in
        the file are extracted.

    shard: tuple of ints, optional, default=None
        A tuple (index, count) representing worker_id and num_workers
        count. Necessary to evenly split/shard the dataset among many
        workers (i.e. >1).

    compression_type: str, optional, default=None
        The type of compression used for the tfrecord. Choose either
        'gzip' or None.

    Yields:
    -------
    features: dict of {str, value}
        Decoded bytes of the features into its respective data type (for
        an individual record). `value` is either going to be an np.ndarray
        in the instance of an `Example` and a list of np.ndarray in the
        instance of a `SequenceExample`.
    """
    return ExampleIterator(  # type: ignore
        data_path=data_path,
        index=index,
        description=description,
        shard=shard,
        clip=clip,
        compression_type=compression_type,
        datum_bytes=datum_bytes
    )


class TFRecordDataset(torch.utils.data.IterableDataset):
    """Parse (generic) TFRecords dataset into `IterableDataset` object,
    which contain `np.ndarrays`s. By default it treats the TFRecords as containing `tf.Example`.
    Otherwise, it assumes it is a `tf.SequenceExample`.

    Params:
    -------
    data_path: str
        The path to the tfrecords file.

    index_path: str or None
        The path to the index file.

    description: list or dict of str, optional, default=None
        List of keys or dict of (key, value) pairs to extract from each
        record. The keys represent the name of the features and the
        values ("byte", "float", or "int") correspond to the data type.
        If dtypes are provided, then they are verified against the
        inferred type for compatibility purposes. If None (default),
        then all features contained in the file are extracted.

    shuffle_queue_size: int, optional, default=None
        Length of buffer. Determines how many records are queued to
        sample from.

    transform : a callable, default = None
        A function that takes in the input `features` i.e the dict
        provided in the description, transforms it and returns a
        desirable output.

    compression_type: str, optional, default=None
        The type of compression used for the tfrecord. Choose either
        'gzip' or None.

    """

    def __init__(
        self,
        data_path: str,
        index_path: Union[str, None] = None,
        description: Union[List[str], Dict[str, str], None] = None,
        shuffle_queue_size: Optional[int] = None,
        transform: Callable[[dict], Any] = None,
        compression_type: Optional[str] = None,
        autoshard: bool = False,
        clip: Optional[int] = None,
    ) -> None:
        super(TFRecordDataset, self).__init__()
        self.data_path = data_path
        self.index_path = index_path
        self.description = description
        self.shuffle_queue_size = shuffle_queue_size
        self.transform = transform or (lambda x: x)
        self.compression_type = compression_type
        self.autoshard = autoshard
        self.clip = clip

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if self.autoshard and worker_info is not None:
            shard = worker_info.id, worker_info.num_workers
            np.random.seed(worker_info.seed % np.iinfo(np.uint32).max)
        else:
            shard = None
        it = iter(tfrecord_loader(
            data_path=self.data_path,
            index=self.index_path,
            description=self.description,
            shard=shard,
            clip=self.clip,
            compression_type=self.compression_type)
        )
        if self.transform:
            it = map(self.transform, it)
        return it


class PancanTFRecordDataset(TFRecordDataset):
    def __init__(self, tfrecords_path, index_path, transform):
        self.index = np.load(index_path)['arr_0']
        super().__init__(tfrecords_path, self.index, transform=transform)

    def __len__(self):
        return len(self.index)
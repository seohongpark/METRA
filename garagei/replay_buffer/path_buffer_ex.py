"""A replay buffer that efficiently stores and can sample whole paths."""
import collections

import numpy as np


class PathBufferEx:
    """A replay buffer that stores and can sample whole paths.

    This buffer only stores valid steps, and doesn't require paths to
    have a maximum length.

    Args:
        capacity_in_transitions (int): Total memory allocated for the buffer.

    """

    def __init__(self, capacity_in_transitions, pixel_shape):
        self._capacity = capacity_in_transitions
        self._transitions_stored = 0
        self._first_idx_of_next_path = 0
        # Each path in the buffer has a tuple of two ranges in
        # self._path_segments. If the path is stored in a single contiguous
        # region of the buffer, the second range will be range(0, 0).
        # The "left" side of the deque contains the oldest path.
        self._path_segments = collections.deque()
        self._buffer = {}

        if pixel_shape is not None:
            self._pixel_dim = np.prod(pixel_shape)
        else:
            self._pixel_dim = None
        self._pixel_keys = ['obs', 'next_obs']

    def add_path(self, path):
        """Add a path to the buffer.

        Args:
            path (dict): A dict of array of shape (path_len, flat_dim).

        Raises:
            ValueError: If a key is missing from path or path has wrong shape.

        """
        path_len = self._get_path_length(path)
        first_seg, second_seg = self._next_path_segments(path_len)
        # Remove paths which will overlap with this one.
        while (self._path_segments and self._segments_overlap(
                first_seg, self._path_segments[0][0])):
            self._path_segments.popleft()
        while (self._path_segments and self._segments_overlap(
                second_seg, self._path_segments[0][0])):
            self._path_segments.popleft()
        self._path_segments.append((first_seg, second_seg))
        for key, array in path.items():
            if self._pixel_dim is not None and key in self._pixel_keys:
                pixel_key = f'{key}_pixel'
                state_key = f'{key}_state'
                if pixel_key not in self._buffer:
                    self._buffer[pixel_key] = np.random.randint(0, 255, (self._capacity, self._pixel_dim), dtype=np.uint8)  # For memory preallocation
                    self._buffer[state_key] = np.zeros((self._capacity, array.shape[1] - self._pixel_dim), dtype=array.dtype)
                self._buffer[pixel_key][first_seg.start:first_seg.stop] = array[:len(first_seg), :self._pixel_dim]
                self._buffer[state_key][first_seg.start:first_seg.stop] = array[:len(first_seg), self._pixel_dim:]
                self._buffer[pixel_key][second_seg.start:second_seg.stop] = array[len(first_seg):, :self._pixel_dim]
                self._buffer[state_key][second_seg.start:second_seg.stop] = array[len(first_seg):, self._pixel_dim:]
            else:
                buf_arr = self._get_or_allocate_key(key, array)
                buf_arr[first_seg.start:first_seg.stop] = array[:len(first_seg)]
                buf_arr[second_seg.start:second_seg.stop] = array[len(first_seg):]
        if second_seg.stop != 0:
            self._first_idx_of_next_path = second_seg.stop
        else:
            self._first_idx_of_next_path = first_seg.stop
        self._transitions_stored = min(self._capacity,
                                       self._transitions_stored + path_len)

    def sample_transitions(self, batch_size):
        """Sample a batch of transitions from the buffer.

        Args:
            batch_size (int): Number of transitions to sample.

        Returns:
            dict: A dict of arrays of shape (batch_size, flat_dim).

        """
        idx = np.random.choice(self._transitions_stored, batch_size)
        if self._pixel_dim is not None:
            ret_dict = {}
            keys = set(self._buffer.keys())
            for key in self._pixel_keys:
                pixel_key = f'{key}_pixel'
                state_key = f'{key}_state'
                keys.remove(pixel_key)
                keys.remove(state_key)
                if self._buffer[state_key].shape[1] != 0:
                    ret_dict[key] = np.concatenate([self._buffer[pixel_key][idx], self._buffer[state_key][idx]], axis=1)
                else:
                    ret_dict[key] = self._buffer[pixel_key][idx]
            for key in keys:
                ret_dict[key] = self._buffer[key][idx]
            return ret_dict
        else:
            return {key: buf_arr[idx] for key, buf_arr in self._buffer.items()}

    def _next_path_segments(self, n_indices):
        """Compute where the next path should be stored.

        Args:
            n_indices (int): Path length.

        Returns:
            tuple: Lists of indices where path should be stored.

        Raises:
            ValueError: If path length is greater than the size of buffer.

        """
        if n_indices > self._capacity:
            raise ValueError('Path is too long to store in buffer.')
        start = self._first_idx_of_next_path
        end = start + n_indices
        if end > self._capacity:
            second_end = end - self._capacity
            return (range(start, self._capacity), range(0, second_end))
        else:
            return (range(start, end), range(0, 0))

    def _get_or_allocate_key(self, key, array):
        """Get or allocate key in the buffer.

        Args:
            key (str): Key in buffer.
            array (numpy.ndarray): Array corresponding to key.

        Returns:
            numpy.ndarray: A NumPy array corresponding to key in the buffer.

        """
        buf_arr = self._buffer.get(key, None)
        if buf_arr is None:
            buf_arr = np.zeros((self._capacity, array.shape[1]), array.dtype)
            self._buffer[key] = buf_arr
        return buf_arr

    def clear(self):
        """Clear buffer."""
        self._transitions_stored = 0
        self._first_idx_of_next_path = 0
        self._path_segments.clear()
        self._buffer.clear()

    @staticmethod
    def _get_path_length(path):
        """Get path length.

        Args:
            path (dict): Path.

        Returns:
            length: Path length.

        Raises:
            ValueError: If path is empty or has inconsistent lengths.

        """
        length_key = None
        length = None
        for key, value in path.items():
            if length is None:
                length = len(value)
                length_key = key
            elif len(value) != length:
                raise ValueError('path has inconsistent lengths between '
                                 '{!r} and {!r}.'.format(length_key, key))
        if not length:
            raise ValueError('Nothing in path')
        return length

    @staticmethod
    def _segments_overlap(seg_a, seg_b):
        """Compute if two segments overlap.

        Args:
            seg_a (range): List of indices of the first segment.
            seg_b (range): List of indices of the second segment.

        Returns:
            bool: True iff the input ranges overlap at at least one index.

        """
        # Empty segments never overlap.
        if not seg_a or not seg_b:
            return False
        first = seg_a
        second = seg_b
        if seg_b.start < seg_a.start:
            first, second = seg_b, seg_a
        assert first.start <= second.start
        return first.stop > second.start

    @property
    def n_transitions_stored(self):
        """Return the size of the replay buffer.

        Returns:
            int: Size of the current replay buffer.

        """
        return int(self._transitions_stored)

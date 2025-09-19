import h5py
import numpy as np
#import open3d
import os

class IO:
    @classmethod
    def get(cls, file_path):
        _, file_extension = os.path.splitext(file_path)

        if file_extension in ['.npy']:
            return cls._read_npy(file_path)
        elif file_extension in ['.pcd', '.ply']:
            return cls._read_pcd(file_path)
        elif file_extension in ['.h5']:
            return cls._read_h5(file_path)
        elif file_extension in ['.txt']:
            return cls._read_txt(file_path)
        else:
            raise Exception('Unsupported file extension: %s' % file_extension)

    # References: https://github.com/numpy/numpy/blob/master/numpy/lib/format.py
    @classmethod
    def _read_npy(cls, file_path):
        return np.load(file_path)
       
    # References: https://github.com/dimatura/pypcd/blob/master/pypcd/pypcd.py#L275
    # Support PCD files without compression ONLY!

    @classmethod
    def _read_pcd(cls,file_path):
        """
        Reads a PCD file (ASCII or binary) without Open3D.
        Handles encoding issues and supports both xyz and xyzrgb formats.
        
        Args:
            file_path (str): Path to .pcd file
        
        Returns:
            np.ndarray: Nx3 or Nx6 array of points (and optionally colors)
        
        Raises:
            ValueError: If unsupported PCD format is encountered
        """
        with open(file_path, 'rb') as f:  # Note 'rb' for binary mode
            header = []
            while True:
                line = f.readline().decode('ascii', errors='ignore').strip()
                header.append(line)
                if line.startswith('DATA'):
                    break
            
            # Parse header
            data_format = None
            width, height, fields, size, count = 0, 0, [], [], []
            for line in header:
                if line.startswith('FIELDS'):
                    fields = line.split()[1:]
                elif line.startswith('SIZE'):
                    size = list(map(int, line.split()[1:]))
                elif line.startswith('COUNT'):
                    count = list(map(int, line.split()[1:]))
                elif line.startswith('WIDTH'):
                    width = int(line.split()[1])
                elif line.startswith('HEIGHT'):
                    height = int(line.split()[1])
                elif line.startswith('DATA'):
                    data_format = line.split()[1]
            
            # Verify we have x,y,z fields
            if not all(f in fields for f in ['x', 'y', 'z']):
                raise ValueError("PCD file must contain x,y,z fields")
            
            # Calculate point size in bytes
            point_size = sum(s*c for s,c in zip(size, count))
            num_points = width * height
            
            # Read data
            if data_format == 'ascii':
                data = f.read().decode('ascii')
                points = np.fromstring(data, sep=' ', dtype=np.float32)
                points = points.reshape((num_points, -1))
                # Extract xyz (first 3 columns)
                return points[:, :3]
            
            elif data_format == 'binary':
                data = f.read(point_size * num_points)
                points = np.frombuffer(data, dtype=np.float32)
                points = points.reshape((num_points, -1))
                return points[:, :3]
            
            else:
                raise ValueError(f"Unsupported PCD data format: {data_format}")

    @classmethod
    def _read_txt(cls, file_path):
        return np.loadtxt(file_path)

    @classmethod
    def _read_h5(cls, file_path):
        f = h5py.File(file_path, 'r')
        return f['data'][()]
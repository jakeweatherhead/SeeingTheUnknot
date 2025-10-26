# The MIT License (MIT)
# Copyright © 2025 Jake Weatherhead

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import PIL
from torchvision import transforms
from torch.utils.data import Dataset

class KnotDataset(Dataset):
    """A PyTorch Dataset of two-dimensional knot diagrams and corresponding labels."""
    def __init__(
        self, 
        image_paths: list[str], 
        labels: list[int], 
        transform: transforms.Compose = None
    ):
        self.image_paths: list[str] = image_paths
        self.labels: list[int] = labels
        self.transform: transforms.Compose = transform

    def __len__(
        self
    ) -> int:
        return len(self.image_paths)

    def __getitem__(
        self, 
        index: int
    ) -> tuple[PIL.Image, int]:
        """Loads and returns the image-label pair at the specified index.
        Applies the configured transforms if provided.
        """
        image_path = self.image_paths[index]
        try:
            image = PIL.Image.open(image_path).convert('RGB')
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Image file not found at {image_path}"
            )
        except Exception as e:
            raise Exception(
                f"Error loading image {image_path}: {e}"
            )

        if self.transform:
            image = self.transform(image)

        return image, self.labels[index]
"""Compatibility wrapper for the refactored PackCNN package."""

from pack.bsgs import *
from pack.config import *
from pack.conv import *
from pack.crypto import *
from pack.data import *
from pack.encoding import *
from pack.model import *
from pack.pipeline import batch_CNN, main


if __name__ == "__main__":
    main()

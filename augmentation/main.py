import sys
import config
from augmentation import run_augmentation
from generation import run_generation

match sys.argv[1] if len(sys.argv) > 1 else "":
    case "augmentation":
        run_augmentation(config.OPENAI_REQUEST_DELAY)

    case "generation":
        run_generation(config.OPENAI_REQUEST_DELAY)

    case _:
        print("Invalid command! Supported commands: augmentation, generation")
        sys.exit(1)

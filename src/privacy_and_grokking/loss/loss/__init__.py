from typing import Annotated

from pydantic import Field

from privacy_and_grokking.loss.loss.ce import CrossEntropyLossConfig
from privacy_and_grokking.loss.loss.mse import MSELossConfig

Loss = Annotated[MSELossConfig | CrossEntropyLossConfig, Field(discriminator="name")]

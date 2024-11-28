import dataclasses
from enum import auto, Enum
from typing import List


class SeparatorStyle(Enum):
    """Different separator style."""
    SINGLE = auto()
    TWO = auto()


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""
    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "###"
    sep2: str = None
    version: str = "Unknown"
    skip_next: bool = False

    def get_prompt(self):
        messages = self.messages
        # sep, sep2 = self.sep, self.sep2
        # ret = self.system + sep
        # for i, (role, message) in enumerate(messages):
        #     if i % 2 == 0:
        #         current_sep = sep
        #     else:
        #         current_sep = sep2
        #     ret += f"{role}: {message}{current_sep}"

        seps = [self.sep, self.sep2]
        ret = self.system + seps[0]
        for i, (role, message) in enumerate(messages):
            if message:
                ret += role + ": " + message + seps[i % 2]  # TODO: 这边的seps会加入一个终止符，终止符是每个对话最后加，还是所有对话完毕之后加？
            else:
                ret += role + ":"

        return ret



    def append_message(self, role, message):
        self.messages.append([role, message])

    def copy(self):
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            version=self.version
        )

    def dict(self):
        return {
            "system": self.system,
            "roles": self.roles,
            "messages": [[x, y[0] if type(y) is tuple else y] for x, y in self.messages],
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
        }


conv_vicuna_v1 = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions.",
    roles=("USER", "ASSISTANT"),
    version="v1",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)

default_conversation = conv_vicuna_v1
conv_templates = {"v1": conv_vicuna_v1,}

if __name__ == "__main__":
    print(conv_vicuna_v1.get_prompt())

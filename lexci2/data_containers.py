"""Containers for the data that is handled by LExCI.

File:   lexci2/data_containers.py
Author: Kevin Badalian (badalian_k@mmp.rwth-aachen.de)
        Teaching and Research Area Mechatronics in Mobile Propulsion (MMP)
        RWTH Aachen University
Date:   2022-04-06


Copyright 2023 Teaching and Research Area Mechatronics in Mobile Propulsion,
               RWTH Aachen University

Licensed under the Apache License, Version 2.0 (the "License"); you may not use
this file except in compliance with the License. You may obtain a copy of the
License at: http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
"""

import copy
import json
import numpy as np
from typing import Any, Optional, Iterator


class Experience:
    """Container for a single experience."""

    def __init__(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        new_obs: np.ndarray,
        reward: float,
        done: bool = False,
        t: Optional[float] = None,
        aux_data: dict[str, Any] = {},
    ) -> None:
        """Initialize the experience.

        Arguments:
            - obs: np.ndarray
                  An observed state.
            - action: np.ndarray
                  The action an agent chose based on the current observation.
            - new_obs: np.ndarray:
                  New observed state as a result of taking the chosen action.
            - reward: float
                  The reward that was given by the environment for the chosen
                  action based on the observation.
            - done: bool (Default: False)
                  Whether the agent reached a terminal state with this
                  experience.
            - t: Optional[float] (Default: None)
                  Time since the start of the episode this experience belongs
                  to.
            - aux_data: dict[str, Any] (Default: {})
                  Dictionary containing additional auxiliary data.
        """

        self.obs = copy.deepcopy(obs)
        self.action = copy.deepcopy(action)
        self.new_obs = copy.deepcopy(new_obs)
        self.reward = reward
        self.done = done
        self.t = t
        self.aux_data = copy.deepcopy(aux_data)

    @classmethod
    def from_json(cls, json_string: str) -> "Experience":
        """Create an experience container based on a JSON string.

        Arguments:
            - json_string: str
                  A JSON string representation of an experience.

        Returns:
            - _: Experience
                  An experience container.

        Raises:
            - ValueError
        """

        try:
            d = json.loads(json_string)
            d["obs"] = np.array(d["obs"], dtype=np.float32)
            d["action"] = np.array(d["action"], dtype=np.float32)
            d["new_obs"] = np.array(d["new_obs"], dtype=np.float32)
            return cls(
                d["obs"],
                d["action"],
                d["new_obs"],
                d["reward"],
                d["done"],
                d["t"],
                d["aux_data"],
            )
        except:
            raise ValueError

    def to_json(self, indent_level: Optional[int] = None) -> str:
        """Translate the experience into a JSON string.

        Arguments:
            - indent_level: Optional[int]
                  Number of whitespaces to use for indentation.

        Returns:
            - _: str
                  A JSON string.
        """

        return json.dumps(self, cls=JsonEncoder, indent=indent_level)

    def __str__(self) -> str:
        """Get a string representation of the experience.

        Returns:
            - _: str
                  A string representation of the experience.
        """

        s = (
            f"[obs={self.obs}, action={self.action}, new_obs={self.new_obs},"
            + f" reward={self.reward}, done={self.done}, t={self.t}"
        )
        for k, v in self.aux_data.items():
            s += f", {k}={v}"
        s += "]"
        return s


class Episode:
    """Container for a LExCI episode."""

    def __init__(self, agent_id: str, exps: list[Experience] = []) -> None:
        """Initialize the episode.

        Arguments:
            - agent_id: str
                  ID of the agent that generated this episode.
            - exps: list[Experience] (Default: [])
                  (Initial) list of experiences.
        """

        self.agent_id = agent_id
        self.exps = copy.deepcopy(exps)

    @classmethod
    def from_json(cls, json_string: str) -> "Episode":
        """Create an episode container based on a JSON string.

        Arguments:
            - json_string: str
                  A JSON string representation of an episode.

        Returns:
            - _: Episode
                  An episode container.

        Raises:
            - ValueError
        """

        try:
            d = json.loads(json_string)
            exps = []
            for e in d["exps"]:
                exps.append(Experience.from_json(json.dumps(e)))
            return cls(d["agent_id"], exps)
        except:
            raise ValueError

    def to_json(self, indent_level: Optional[int] = None) -> str:
        """Translate the episode into a JSON string.

        Arguments:
            - indent_level: Optional[int] (Default: None)
                  Number of whitespaces to use for indentation.

        Returns:
            - _: str
                  A JSON string.
        """

        return json.dumps(self, cls=JsonEncoder, indent=indent_level)

    def __str__(self) -> str:
        """Get a string representation of the episode.

        Returns:
            - _: str
                  A string representation of the episode.
        """

        s = f"[Agent ID: {self.agent_id}"
        for exp in self.exps:
            s += f"\n  {exp}"
        s += "\n]"
        return s

    def __len__(self) -> int:
        """Get the number of experiences in the episode.

        Returns:
            - _: int
                  Number of experiences.
        """

        return len(self.exps)

    def __iter__(self) -> Iterator[Experience]:
        """Get an iterator for the episode's experiences.

        Returns:
            - _: Iterator[Experience]
                  An iterator.
        """

        return iter(self.exps)

    def __getitem__(self, i: int) -> Experience:
        """Get the `(i + 1)`-th experience in the episode.

        Arguments:
            - i: int
                  An index.

        Returns:
            - _: Experience
                  Experience with index `i`.
        """

        return self.exps[i]

    def append_experience(self, exp: Experience) -> None:
        """Append an experience to the episode.

        Arguments:
            - exp: Experience
                  An experience.
        """

        self.exps.append(copy.deepcopy(exp))

    def append_experiences(self, exps: list[Experience]) -> None:
        """Append a list of experiences to the episode.

        Arguments:
            - exps: list[Experience]
                  The experiences to append.
        """

        self.exps.extend(copy.deepcopy(exps))

    def get_num_experiences(self) -> int:
        """Get the (current) number of experiences in the LExCI episode.

        Returns:
            - _: int
                  Number of experiences in the episode.
        """

        return len(self.exps)


class Cycle:
    """Container for data generated during a LExCI cycle."""

    def __init__(self, eps: list[Episode] = []) -> None:
        """Initialize the container.

        Arguments:
            - eps: list[Episode] (Default: None)
                  (Initial) list of episodes
        """

        self.eps = copy.deepcopy(eps)

    @classmethod
    def from_json(cls, json_string: str) -> "Cycle":
        """Create a cycle container based on a JSON string.

        Arguments:
            - json_string: str
                  A JSON string representation of some cycle data.

        Returns:
            - _: CycleData
                  A cycle container.

        Raises:
            - ValueError
        """

        try:
            l = json.loads(json_string)
            eps = []
            for e in l:
                eps.append(Episode.from_json(json.dumps(e)))
            return cls(eps)
        except:
            raise ValueError

    def to_json(self, indent_level: Optional[int] = None) -> str:
        """Translate the cycle into a JSON string.

        Arguments:
            - indent_level: Optional[int] (Default: None)
                  Number of whitespaces to use for indentation.

        Returns:
            - _: str
                  A JSON string.
        """

        return json.dumps(self, cls=JsonEncoder, indent=indent_level)

    def __str__(self) -> str:
        """Get a string representation of the cycle.

        Returns:
            - _: str
                  A string representation of the cycle.
        """

        s = f"["
        for eps in self.eps:
            eps_str = str(eps).replace("\n", "\n  ")
            s += f"\n  {eps_str}"
        s += "\n]"
        return s

    def __len__(self) -> int:
        """Get the number of episodes in the cycle.

        Returns:
            - _: int
                  Number of episodes in the data container.
        """

        return len(self.eps)

    def __iter__(self) -> Iterator[Episode]:
        """Get an iterator for this cycle's episodes.

        Returns:
            - _: Iterator[Episode]
                  An iterator.
        """

        return iter(self.eps)

    def __getiitem__(self, i: int) -> Episode:
        """Get the `(i + 1)`-th episode in the cycle.

        Arguments:
            - i: int
                  An index.

        Returns:
            - _: Episode
                  Episode with index `i`.
        """

        return self.eps[i]

    def add_episode(self, eps: Episode) -> None:
        """Add an episode to the cycle.

        Arguments:
            - eps: Episode
                  An episode.
        """

        self.eps.append(copy.deepcopy(eps))

    def get_num_episodes(self) -> int:
        """Get the (current) number of episodes in the LExCI cycle.

        Returns:
            - _: int
                  Number of episodes in the data container.
        """

        return len(self.eps)

    def get_num_experiences(self) -> int:
        """Get the (current) number of experiences in the LExCI cycle.

        Returns:
            - _: int
                  Number of experiences in the cycle.
        """

        num_experiences = 0
        for e in self.eps:
            num_experiences += e.get_num_experiences()
        return num_experiences


class JsonEncoder(json.JSONEncoder):
    """JSON encoder for LExCI's data containers."""

    @staticmethod
    def make_dict_serializable(dictionary: dict) -> dict:
        """Make a dictionary JSON-serializable by converting non-standard data
        types to string. Nested dictionaries or lists are not considered.

        Arguments:
            - dictionary: dict
                  A dictionary. The original remains unchanged.

        Returns:
            - _: dict
                  A copy of the dictionary where anything that isn't
                  JSON-serializable has been converted to string.
        """

        # Create a copy of the input dictionary so that isn't changed by this
        # method
        d = copy.deepcopy(dictionary)

        for k, v in d.items():
            try:
                # Check whether, in theory, the value could be serialized
                json.dumps(v)
            except:
                if type(v) == np.ndarray:
                    d[k] = v.tolist()
                else:
                    d[k] = str(v)

        # Return the cleaned dictionary
        return d

    def default(self, obj: Any) -> dict[str, Any]:
        """Convert an object into a JSON-serializable entity.

        Arguments:
            - obj: Any
                  Object to convert.

        Returns:
            - _: Any
                  A JSON-serializable object.
        """

        if isinstance(obj, Experience):
            d = {
                "obs": obj.obs.tolist(),
                "action": obj.action.tolist(),
                "new_obs": obj.new_obs.tolist(),
                "reward": obj.reward,
                "done": obj.done,
                "t": obj.t,
                "aux_data": JsonEncoder.make_dict_serializable(obj.aux_data),
            }
            return d
        elif isinstance(obj, Episode):
            d = {"agent_id": obj.agent_id, "exps": obj.exps}
            return d
        elif isinstance(obj, Cycle):
            return obj.eps
        else:
            return json.JSONEncoder.default(self, obj)

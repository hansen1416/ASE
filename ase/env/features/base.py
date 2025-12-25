# env/features/base.py
from __future__ import annotations
from abc import ABC

class Feature(ABC):
    """Simple no-op hook base class."""
    enabled: bool = True

    def on_create_envs(self, task, num_envs: int) -> None:
        pass

    def on_humanoid_actor_created(self, task, env_id: int, env_ptr) -> None:
        pass

    def on_post_init_tensors(self, task) -> None:
        """Called after task._root_states / actor ids are ready."""
        pass

    def on_reset_envs(self, task, env_ids) -> None:
        pass

    def on_post_physics_step(self, task) -> None:
        pass

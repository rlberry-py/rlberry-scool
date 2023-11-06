###########
rlberry API
###########

.. currentmodule:: rlberry

Manager
====================

Main classes
--------------------

.. autosummary::
  :toctree: generated/
  :template: class.rst


    manager.ExperimentManager

Evaluation and plot
--------------------

.. autosummary::
   :toctree: generated/
   :template: function.rst

   manager.evaluate_agents
   manager.plot_writer_data


Agents
====================

Agent importation tools
-----------------------

.. autosummary::
   :toctree: generated/
   :template: class.rst

   agents.stable_baselines.StableBaselinesAgent

Environments
============



Environment tools
-----------------

.. autosummary::
   :toctree: generated/
   :template: function.rst

    envs.gym_make
    envs.atari_make
    envs.PipelineEnv


Seeding
====================

.. autosummary::
   :toctree: generated/
   :template: function.rst

   seeding.safe_reseed
   seeding.set_external_seed

Utilities, Logging & Typing
===========================

Environment Wrappers
====================

.. autosummary::
  :toctree: generated/
  :template: class.rst

  wrappers.discretize_state.DiscretizeStateWrapper
  wrappers.gym_utils.OldGymCompatibilityWrapper
  wrappers.RescaleRewardWrapper
  wrappers.vis2d.Vis2dWrapper
  wrappers.WriterWrapper

Bandits
=======

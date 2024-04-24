.. _api:

###########
rlberry API
###########


Manager
====================

Main classe
-----------

.. autosummary::
  :toctree: generated/
  :template: class.rst

    rlberry.manager.ExperimentManager

Evaluation and plot
--------------------

.. autosummary::
   :toctree: generated/
   :template: function.rst

   rlberry.manager.evaluate_agents
   rlberry.manager.plot_writer_data
   rlberry.manager.read_writer_data
   rlberry.manager.compare_agents

.. autosummary::
  :toctree: generated/
  :template: class.rst

   rlberry.manager.AdastopComparator
   
Agents & Environments
=====================

Basic agents
------------
.. autosummary::
   :toctree: generated/
   :template: class.rst

   rlberry_scool.agents.dynprog.ValueIteration
   rlberry_scool.agents.linear.LSVIUCBAgent
   rlberry_scool.agents.tabular_rl.QLAgent
   rlberry_scool.agents.tabular_rl.SARSAAgent

Basic environments
------------------
.. autosummary::
   :toctree: generated/
   :template: class.rst

    rlberry_scool.envs.GridWorld
    rlberry_scool.envs.Chain
   
Agent importation tools
-----------------------

.. autosummary::
   :toctree: generated/
   :template: class.rst

   rlberry.agents.stable_baselines.StableBaselinesAgent

Environment tools
-----------------

.. autosummary::
   :toctree: generated/
   :template: function.rst

    rlberry.envs.gym_make

Seeding
====================

.. autosummary::
   :toctree: generated/
   :template: function.rst

   rlberry.seeding.safe_reseed
   rlberry.seeding.set_external_seed

Environment Wrappers
====================

.. autosummary::
  :toctree: generated/
  :template: class.rst

  rlberry.wrappers.discretize_state.DiscretizeStateWrapper
  rlberry.wrappers.RescaleRewardWrapper
  rlberry.wrappers.WriterWrapper



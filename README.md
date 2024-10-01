# Federated C-MAMs

F

## Dependencies Installation

The Federated C-MAMs project is managed by through the ``pyproject.toml`` file. To install the project, run the following command from the root directory of the project:

```bash
poetry install
```

## How the project is structured

The entire running of each model, dataset, etc. is entirely managed by a YAML configuration file. The expected components in these YAML files change depending on the type of model being run. As of writing, there are four main model types:

1. Regular multimodal
2. Cross-modal Association Models (C-MAMs)
3. Federated multimodal
4. Federated C-MAMs

All model types share a lot of the same components, but the C-MAMs and federated models contain quite a few additional components. The YAML configuration files are stored in the ``configs`` directory. The ``configs`` directory is further divided into subdirectories based on the model type. The ``configs`` directory has the following structure:

```text
configs
├── avmnist
│   ├── avmnist_multimodal.yaml
│   └── cmams
│       ├── avmnist_a_to_i.yaml
│       └── avmnist_i_to_a.yaml
├── mosei
│   ├── cmams
│   │   ├── mosei_cmams_al_to_v.yaml
│   │   ├── mosei_cmams_a_to_l.yaml
│   │   ├── mosei_cmams_a_to_v.yaml
│   │   ├── mosei_cmams_av_to_l.yaml
│   │   ├── mosei_cmams_l_to_a.yaml
│   │   ├── mosei_cmams_l_to_v.yaml
│   │   ├── mosei_cmams_vl_to_a.yaml
│   │   ├── mosei_cmams_v_to_a.yaml
│   │   └── mosei_cmams_v_to_l.yaml
│   ├── federated
│   │   └── mosei_federated_multimodal.yaml
│   └── train_mosei_multimodal.yaml
├── mosei_tsne_config.yaml
└── tsne_config.yaml
```

### Shared Components

The shared components across all model types are:

1. experiment: Creates an instance of the ``ExperimentConfig`` class. Contains information about the experiment, such as the name of the experiment, the run id, seed, device etc.
2. logging: Creates an instances of the ``LoggingConfig`` class. Contains information about where
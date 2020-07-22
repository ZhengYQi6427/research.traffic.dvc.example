# research.traffic.dvc.example

To keep track of which models were trained with which data, we use dvc (Data Version Control, https://dvc.org/) to version the data, similar to versioning and tracking the code and configurations. It approaches data versioning in a similar way to Git.  DVC makes it easier to work on Data projects using stages and pipelines resulting in a significant gain in productivity and collaboration.

We begin with Yolo-darknet, generate dvc stages to build a pipeline in order to version dependencies and compare metrics in concise way.

Code for "Seeing the Unknot"
===========================================================================

This framework was used to fine-tune a CNN and ViT from the Timm library on a
dataset of two-dimensional knot diagrams.

We studied the unknot recognition problem asking whether or not modern computer vision
can determine knot-triviality from knot diagram inputs.

For more details and our findings see: _url-coming-soon_.  

Before running the code, create an .env file with the variables defined in 
src/seeing-the-unknot/.env.example.

Download the data here: https://console.cloud.google.com/storage/browser/knot_data;tab=objects?project=computervisionknottheory&prefix=&forceOnObjectsSortingFiltering=false

Extract the `train.zip`, `val.zip`, and `test.zip` data splits into the `SeeingTheUnknot/knot_data` directory to form:

```text
SeeingTheUnknot/
├─ knot_data/
│  ├─ train/
│  ├─ val/
│  └─ test/
└─ ...
```
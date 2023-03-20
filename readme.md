# Description
This repo contains the processing code to process images with instabilities. An instability can look similar to this:

![instability example](./doc/imgs/insta_example.tif)

Two things are calculted by the porcessing tool:
1. An Area ratio r 
2. Outer shape over time 

## Usage
The procedure implemented to get the ratio can be seen here:

![procedure](./doc/imgs/procedure.svg)

A configuration might lool like this:
```json
{
    "data_path" : [
        "\\\\gssnas",
        "bigdata",
        "FWDT",
        "DFischer",
        "image_proc"
    ],
    "cases" : [
        "0.1mmspacer_0.015ml_2"
    ]
}
```
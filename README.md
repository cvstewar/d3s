# Modified D3S with Kalman Filter for SYDE 673

This is a modified version of D3S for a course project in SYDE 673 at the University of Waterloo.

By: Curtis Stewart

## Installing D3S with the new VOT Python Toolkit

### Clone the repository

```bash
mkdir D3S
git clone https://github.com/cvstewar/d3s.git D3S
```

### Install D3S dependencies

Run the installation script to install all the dependencies. This could be either `install.sh` on linux or `install.bat` on Windows machines. I tested this on a machine running Windows 10. You need to provide the conda install path (e.g. ~/anaconda3) and the name for the created conda environment.

See below for the command that I ran. Note that I made changes to `install.bat` for compatibility. Ensure you read this script over carefully before running it. D3S was made for cuda 9.0 and pytorch 1.1.0 and the `install.bat` will install these versions.

```bash
cd D3S
install.bat C:\ProgramData\Anaconda3 d3s
```

The pre-trained network for the D3S is not part of this repository. You can download it at the link provided below in the original readme from the D3S authors.

This tracker was tested on a Windows 10 machine with a NVIDIA GTX 1070 graphics card and cudatoolkit version 9.0.

### Install the VOT Python Toolkit

See below for installation instructions for the new VOT python toolkit for testing with D3S. These are modified from the linked [VOT Toolkit Installation](https://votchallenge.net/howto/tutorial_python.html) instructions. See the [VOT evaluation toolkit](https://github.com/votchallenge/toolkit) on GitHub for the code behind the toolkit. Also, refer to the [Tracker Integration Examples](https://github.com/votchallenge/integration) on GitHub for more details on how the tracker was integrated. The pytracking/vot.py file was updated with the latest version in the "Tracker Integration Examples" GitHub repo at the time of writing this.

**Note**: To install this toolkit, you will need a C++ compiler. For example, you could install the C++ build tools with Visual Studio at the link below:

[Visual Studio with C++](https://visualstudio.microsoft.com/vs/features/cplusplus/)

Install the VOT Python Toolkit with the command below:

```bash
pip install git+https://github.com/votchallenge/vot-toolkit-python
```

#### NOTE - Fixing Numpy Errors

Note that I had to perform the following in order to fix errors when importing numpy. See the stack overflow page [here](https://stackoverflow.com/questions/58868528/importing-the-numpy-c-extensions-failed) for more details on the issue.

```bash
pip uninstall -y numpy
pip uninstall -y setuptools
pip install setuptools
pip install numpy
```

### Creating a workspace

This will create a workspace folder and download the VOT2018 or VOT2020 dataset sequences into it. This utilizes the VOT toolkit installed earlier and will set up the folder structure to be compatible with the toolkit for evaluation. Note that this assumes you are at the directory level of D3S's parent folder. See details on the VOT2020 dataset [here](https://www.votchallenge.net/vot2020/dataset.html).

```bash
mkdir vot-2018
vot initialize vot2018 --workspace vot-2018
```

Or,

```bash
mkdir vot-2020
vot initialize vot2020 --workspace vot-2020
```

See the folder structure below:

```markdown
-D3S
    -ltr
    -pytracking
    ...
-vot-2018
    -results
    -sequences
    -trackers.ini
    ...
-vot-2020
    -results
    -sequences
    -trackers.ini
    ...
```

### Specifying paths for D3S

See below for instructions on setting up paths for D3S. Note that on Windows, any backslash `\` in the paths should be escaped with `\\` for the path to work.

1.) Specify the path to the D3S [pre-trained segmentation network](http://data.vicos.si/alanl/d3s/SegmNet.pth.tar) by setting the `params.segm_net_path` in the `pytracking/parameters/segm/default_params.py`. <br/>
2.) Specify the path to the VOT 2018 dataset "sequences" folder by setting the `vot18_path` in the `pytracking/evaluation/local.py`. (This is not needed if using the VOT Python Toolkit for evaluation) <br/>
3.) Activate the conda environment

```bash
conda activate d3s
```

### Setting up the trackers.ini

Refer to the [VOT Toolkit Installation](https://votchallenge.net/howto/tutorial_python.html) instructions for more details on how to setup the trackers.ini. This file is located in the vot workspace folder created in the last step. Note that the paths variable here should point to the folder containing the vot_wrapper.py and vot2020wrappper.py files, assuming that the user is in the vot workspace directory created earlier.

See below for an example trackers.ini for testing on VOT2018.

```ini
[D3SPython]  # <tracker-name>
label = PyD3S
protocol = traxpython

command = vot_wrapper

# Specify a path to trax python wrapper if it is not visible (separate by ; if using multiple paths)
paths = ../D3S/pytracking

# Additional environment paths
# env_PATH = <additional-env-paths>;${PATH}
```

For testing on VOT2020, replace `vot_wrapper` with `vot2020wrapper`, otherwise the trackers.ini is the same.

### Testing the tracker

The `vot test` command below can be used to verify the tracker has integrated properly with the VOT toolkit and that there aren't any errors during execution.

```bash
cd vot-2020
vot test D3SPython
```

### Evaluating on the VOT dataset

The `vot evaluate` command is what starts the testing on the dataset. Note that this may take a few hours.

```bash
vot evaluate D3SPython
```

### Analyzing the results

The vot analysis command will compare the tracker results from the evaluate command with the ground truth data to evaluate the tracker performance on the dataset. By default, this produces a .html output file in the analysis subfolder of the workspace. This takes a couple minutes to run.

The `--nocache` flag ensures that it recalculates the analysis results in case `vot evaluate` has been run again after performing analysis earlier. See the argument parser in the [VOT Utilities CLI](https://github.com/votchallenge/toolkit/blob/master/vot/utilities/cli.py) main() function for details on the function calls.

```bash
vot analysis D3SPython --nocache
```

### Visualizing results

I have created a script to read the output of the `vot evaluate` results and produce annotated images and videos with the tracking predictions. To run on VOT2020, first nagivate to the pytracking folder, then run the vot_mask_vis.py script. Note the assumed file structure mentioned in the file. The resulting visualization will be located in the vot-2020 results folder in a new folder called "visualization" under the tracker.

```bash
cd pytracking
python vot_mask_vis.py
```

---

## D3S - A Discriminative Single Shot Segmentation Tracker [CVPR2020]

**See below for the original readme from the D3S authors.**

Python (PyTorch) implementation of the D3S tracker, presented at CVPR 2020.

## Publication

Alan Lukežič, Jiří Matas and Matej Kristan.
<b>D3S - A Discriminative Single Shot Segmentation Tracker.</b>
<i>IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) 2020</i>.</br>
[Paper](https://arxiv.org/abs/1911.08862) </br>

<b>BibTex citation:</b></br>
@InProceedings{Lukezic_CVPR_2020,<br>
Title = {D3S - A Discriminative Single Shot Segmentation Tracker},<br>
Author = {Lukezic, Alan and Matas, Jiri and Kristan, Matej},<br>
Booktitle = {CVPR},<br>
Year = {2020}<br>
}

## Summary of the D3S tracker

Template-based discriminative trackers are currently the dominant tracking paradigm due to their robustness, but are restricted to bounding box tracking and a limited range of transformation models, which reduces their localization accuracy. We propose a discriminative single-shot segmentation tracker -- D3S, which narrows the gap between visual object tracking and video object segmentation. A single-shot network applies two target models with complementary geometric properties, one invariant to a broad range of transformations, including non-rigid deformations, the other assuming a rigid object to simultaneously achieve high robustness and online target segmentation. Without per-dataset finetuning and trained only for segmentation as the primary output, D3S outperforms all trackers on VOT2016, VOT2018 and GOT-10k benchmarks and performs close to the  state-of-the-art trackers on the TrackingNet. D3S outperforms the leading segmentation tracker SiamMask on video  object segmentation benchmarks and performs on par with top video object segmentation algorithms, while running an order of magnitude faster, close to real-time.

<p style="width:100%, text-align:center"><a href="url"><img src="https://raw.githubusercontent.com/alanlukezic/d3s/master/pytracking/utils/d3s-architecture.png" width="640"></a></p>

## Installation

### Clone the GIT repository

```bash
git clone https://github.com/alanlukezic/d3s.git .
```

### Install dependencies

Run the installation script to install all the dependencies. You need to provide the conda install path (e.g. ~/anaconda3) and the name for the created conda environment (here ```pytracking```).  

```bash
bash install.sh conda_install_path pytracking
```

To install the dependencies on a Windows machine, use the `install.bat` script.
The pre-trained network for the D3S is not part of this repository. You can download it [here](http://data.vicos.si/alanl/d3s/SegmNet.pth.tar).

The tracker was tested on the Ubuntu 16.04 machine with a NVidia GTX 1080 graphics card and cudatoolkit version 9.
It was tested on Window 10 as well, but network training is tested on Linux only.

### Test the tracker

1.) Specify the path to the D3S [pre-trained segmentation network](http://data.vicos.si/alanl/d3s/SegmNet.pth.tar) by setting the `params.segm_net_path` in the `pytracking/parameters/segm/default_params.py`. <br/>
2.) Specify the path to the VOT 2018 dataset by setting the `vot18_path` in the `pytracking/evaluation/local.py`. <br/>
3.) Activate the conda environment

```bash
conda activate pytracking
```

4.) Run the script pytracking/run_tracker.py to run D3S using VOT18 sequences.  

```bash
cd pytracking
python run_tracker.py segm default_params --dataset vot18 --sequence <seq_name> --debug 1
```

### Evaluate the tracker using VOT

We provide a VOT Matlab toolkit integration for the D3S tracker. There is the `tracker_D3S.m` Matlab file in the `pytracking/utils`, which can be connected with the toolkit. It uses the `vot_wrapper.py` script to integrate the tracker to the toolkit.

### Training the network

The D3S is pre-trained for segmentation task only on the YouTube VOS dataset. Download the VOS training dataset (2018 version) and copy the files `vos-list-train.txt` and `vos-list-val.txt` from `ltr/data_specs` to the `train` directory of the VOS dataset.

Set the `vos_dir` variable in `ltr/admin/local.py` to the VOS `train` directory on your machine.
Download the bounding boxes from [this link](http://data.vicos.si/alanl/d3s/rectangles.zip) and copy them to the sequence directories.
Run training by running the following command:

```bash
python run_training.py segm segm_default
```

## Pytracking

This is a modified version of the python framework pytracking based on **PyTorch**. We would like to thank the authors Martin Danelljan and Goutam Bhat for providing such a great framework.

## Video

Check out the [video](https://www.youtube.com/watch?v=E3mN_hCRHu0) with tracking and segmentation results of the D3S tracker.

## Contact

* Alan Lukežič (email: alan.lukezic@fri.uni-lj.si)

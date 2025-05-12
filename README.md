# HydraAbsorption
 Software for analysing Spectral- and Absortipn-Mapping.<br>
 This Software is for Absorption Maps created by SymphoTime coupled with the HydraLabX1 System.<br>
 It is also designed to work with Spectral Maps from Andor Solis Spectrometer coupled with the HydraLabX1 System.

## Installation & Requirements
Copy the Software to your project Computer.<br>
Requires "sys", "os", "math", normally preinstalled and "PyQt5", "Scipy", "matplotlib" and "numpy".<br>
Else install the requirements.txt: pip install -r /path/to/requirements.txt

## Supports the major Andor Solis features:
- Read the ASC-Files
- Read single spectra
- Read series spectra
- SpecralMaps

## Supports the major SymphoTime features:
- Read the DAT-Files
- Read single images
- Read series images
- AbsorptionMaps
- Single sprectra from AbsorptionMaps

## Usage
Start the Software from Python. 

**Absorption Mapping**  
The function converts multiple DAT-Files into one large Absorption Map. The files should be named with the individual wavelength used.

**Spectra from Absortpion Map**  
The function takes single Spectra from the previously created large Absorption Map. The User is able to select multiple Positions on the Map to create Spectra.

**PL Spectral Mapping**  
The function takes a series of Spectra from the Andor Solis Software to creat one large Spectral Map. The User is able to select multiple Positions on the Map to extract the single Spectra.<br><br>
For further information, please refere to the Usermanual.



## Extra
In the repo you will find the "Testdata" Folder with some Examples from our Lab for Usage and Testing.

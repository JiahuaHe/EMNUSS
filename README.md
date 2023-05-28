# EMNUSS
A deep learning framework for secondary structure annotation in cryo-EM maps  
Copyright (C) 2020 Jiahua He

Software requirements:

	Python  (https://www.python.org) (ver. 3.7 or later)
	
Python package requirements:

	pytorch (https://pytorch.org) (ver. 1.2 or later)
	mrcfile (https://github.com/ccpem/mrcfile)
	numpy   (https://www.numpy.org)
	tqdm    (https://github.com/tqdm/tqdm)
	
	
The interpolation program "interp3d.f90" should be built as a python package 'interp3d' through **f2py**.

                f2py -c interp3d.f90 -m interp3d
		
This command will generate an ELF file with name like "interp3d.cpython-\*.so". Please keep "interp3d.cpython-\*.so" with all python scripts "\*.py" in the same directory.


In order to run python scripts properly, users should set the python path using one of the following ways.
1. adding python path to the header of each python script like this:

                "#!/path/to/your/python"

2. running the scripts with the full python path like this:

                /path/to/your/python ./emnuss.py ...
	
	
Required files:

	"emd_*.map": EM density map in MRC2014 format (download from EMDB).
	"config.json": config file in JSON format. See "./6MRC/config.json".
	
The compressed trained models are stored in directory "./save/". Unzip them before use.


High resolution Example: EMD-9195

	cd 6MRC/
	../emnuss.py -mi emd_9195.map -mo sspred.mrc -t 0.45 --output

Middle resolution Example: EMD-3329

	cd 5FVM/
	../emnuss.py -mi emd_3329.map -mo sspred.mrc -t 0.044 --output


Attentions: a threshold value should be provided after flag "-t". 
For better visualization, please use the author recommended contour level (or half of it).

Output files:
	"sspred.mrc": secondary structure prediction of the given map (in MRC2014 format).
	"helix.mrc" and "sheet.mrc" and "coil.mrc": predictions of 3 secondary structure classes in individual MRC2014 files (use flag "--output").

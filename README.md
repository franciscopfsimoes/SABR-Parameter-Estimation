# SABR Parameter estimator

This project was developed in the scope of my master's thesis. I would advise to look into the thesis for any technical detail. 

## Installation

Use [pip](https://pip.pypa.io/en/stable/) to install the project's requirements either in your machine or look into create a python venv.

```bash
pip install -r requirements.txt
```

## Usage
To run the project
```
cd src
python __main__.py
````

## Contributing
Pull requests are welcome. The project is still under development by me.

## License
[MIT](https://choosealicense.com/licenses/mit/)


## Brief description of
All principal functions are located inside the __main__.py file.

The Data.txt file consists of only European Put contracts of day 12-12-2019 and maturity 15-12-2019. 
It is used as training data for the SABR parameter estimating algorithm.

In Black.py there are the algorithms for Black-Scholes' contract pricing (European non-dividend puts and calls) and implied volatility.

In SABR.py there are the algorithms for computing SABR implied volatility.

(ExcelDate.py and DataPreProcessing.py are auxiliary files and are specific for preprocessing the Data.txt format)

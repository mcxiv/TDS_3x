# TDS_3x

- [TDS\_3x](#tds_3x)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Documentation](#documentation)
  - [Contributing](#contributing)
  - [License](#license)

**TDS_3x** is an open source library to remotely control some Oscilloscopes from **Tektronix**. 

As of now, it only have been tested with the **TDS 3014C** and the **DPO 3014** oscilloscopes from **Tektronix**, but it should work with many other oscilloscopes from the same brand.

## Installation

Install the dependencies from **requirements.txt** :

```bash
python3 -m pip install -r requirements.txt
```

## Usage
```python
from TDS_3x import *

# Quick usage to take a screenshot :

TDSwfm(IP_ADDRESS, 'CH1', '', '0').savefig('test.png')

# Quick usage to show a live feed :

TDSwfm_live(IP_ADDRESS, 'CH1', '', '0')

# Quick usage to get the trigger state :

TDSwfm_trig(IP_ADDRESS)

# Quick usage to get a measurement (Amplitude, in this case):
TDSwfm_meas(IP_ADDRESS, 'AMP')
```

## Documentation

Sphinx docstrings are used to document the code. You can generate the documentation if you want.

## Contributing

Pull requests are welcome.

## License

[MIT](https://choosealicense.com/licenses/mit/)

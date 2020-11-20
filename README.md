# TDS_3x

TDS_3x is an open source library to remotely control some Oscilloscopes from Tektronix.

It has been tested with a TDS 3014C and a DPO 3014.

Suggested Python version : 3.8
See requirements.txt for other libraries. (Also included the version I was working with)

Please, have a look at the documentation https://mcxiv-python.000webhostapp.com/ (Didn't find a way to use sphinx.autodoc and ReadTheDoc, that's why...)



Quick usage to take a screenshot :

TDSwfm('IpAdress', 'CH1', '', '0').savefig('test.png')

Quick usage to show a live feed :

TDSwfm_live('IpAdress', 'CH1', '', '0')

Quick usage to get the trigger state :

TDSwfm_trig('IpAdress')

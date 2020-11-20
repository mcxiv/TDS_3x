import matplotlib.pyplot as plt
import vxi11
import cv2
import numpy as np
from datetime import datetime

def TDSwfm(ip, channel, arg, wtrace):
    """Return the waveform currently displaying on the oscilloscope

    :param ip: IP Address of the oscilloscope
    :type ip: str
    :param channel: Channel selection (*CH1*, *CH2*, *CH3* or *CH4*)
    :type channel: str
    :param arg: *qm* for quick measures, or/and *cursorH* for horizontal cursors, or/and *cursorV* for vertical cursors, or/and *cursorHV* for both type of cursors
    :type arg: str
    :param wtrace: *1* or *0* or 1* to save a *.txt* of the waveform on local directory or *TRACE* directory if it exists. 1* is used to only save a .txt, nothing else. *.txt* can be used later with *TDSwfm_hist()*
    :type wtrace: str
    :return: *waveform*
    :rtype: *pyplot object*
    :Example:
        .. code-block:: python
        
            TDSwfm('192.168.x.x', 'CH1', 'cursorHV', '0').show()
    
        *will show the waveform*
        
        .. code-block:: python
        
            TDSwfm('192.168.x.x', 'CH2', 'qm cursorHV', '1').savefig('TektroPyx.png')
    
        *will save the waveform as "TektroPyx.png" but also as a *.txt* file*
        
        .. image:: ../images/TDS_3x_wfm.png
            :align: center
        .. image:: ../images/TDS_3x_gif.gif
            :align: center

    .. note:: Can only show one channel at a time
    .. warning:: **For EVERY functions, DELAY MODE must be DISABLED !!**
    """
    instr =  vxi11.Instrument(ip)
    EOL = '\n'
    trace = ''

    instr.write("DATa:SOUrce "+ channel + EOL)
    instr.write("DATa:ENCDG ASCIi"+EOL)
    instr.write("DATA:WIDth 1"+EOL)
    instr.write("DATA:START 1"+EOL)
    instr.write("DATA:STOP 10000"+EOL)

    YZE = float(instr.ask("WFMPre:YZEro?"+EOL))
    trace = trace + str(YZE) + '\n'
    YMU = float(instr.ask("WFMPre:YMUlt?"+EOL))
    trace = trace + str(YMU) + '\n'
    YOF = float(instr.ask("WFMPre:YOFf?"+EOL))
    trace = trace + str(YOF) + '\n'
    Vdiv = float(instr.ask(channel + ":scale?"+EOL))
    trace = trace + str(Vdiv) + '\n'
    Tbase = float(instr.ask("HORizontal:MAIn:SCAle?"+EOL))
    trace = trace + str(Tbase) + '\n'
    Zpos = float(instr.ask(channel + ":position?"+EOL))
    trace = trace + str(Zpos) + '\n'
    Hpos = float(instr.ask("HORizontal:TRIGger:POSition?"+EOL)) * 100
    trace = trace + str(Hpos) + '\n'
    wfm_data = instr.ask("CURVe?"+EOL)
    trace = trace + str(wfm_data) + '\n'
    ##########save trace#################
    seed = str(datetime.now()).replace(' ', '_').replace('-', '').replace(':', '').replace('.', '').split('_')[1]
    if wtrace == '1':
        param = open("trace_"+seed+".txt", "w+")
        param.write(trace)
        param.close()
    if wtrace == '1*':
        try:
            param = open("./TRACE/trace_"+seed+".txt", "w+")
            param.write(trace)
            param.close()
        except:
            param = open("trace_" + seed + ".txt", "w+")
            param.write(trace)
            param.close()
        return seed
    #####################################
    wfm_data = wfm_data.split(',')

    y = []
    x = []
    i = 1
    plt.clf()

    for point in wfm_data:
        y.append((YZE + (YMU*(int(point)-YOF)))+(Zpos*Vdiv))
        x.append(i)
        i+=1

    if 'qm' in arg:
        ###############design qm + qm############
        instr.write('measurement:immed:source ' + channel)
        instr.write('measurement:immed:type rms')
        rms = float(instr.ask('measurement:immed:value?'+EOL))
        instr.write('measurement:immed:type amplitude')
        amp = float(instr.ask('measurement:immed:value?'+EOL))
        instr.write('measurement:immed:type mean')
        mean = float(instr.ask('measurement:immed:value?'+EOL))
        instr.write('measurement:immed:type frequency')
        freq = float(instr.ask('measurement:immed:value?'+EOL))
        instr.write('measurement:immed:type pduty')
        rcy = float(instr.ask('measurement:immed:value?'+EOL))
        plt.scatter(5000, 0, c = 'black', marker = 'o', s = 1, label = 'RMS : ' + str(rms))
        plt.scatter(5000, 0, c = 'black', marker = 'o', s = 1, label = 'AMP : ' + str(amp))
        plt.scatter(5000, 0, c = 'black', marker = 'o', s = 1, label = 'MEAN : ' + str(mean))
        plt.scatter(5000, 0, c = 'black', marker = 'o', s = 1, label = 'FREQ : ' + str(freq))
        plt.scatter(5000, 0, c = 'black', marker = 'o', s = 1, label = 'DCY : ' + str(rcy))
        plt.legend()
    #################cursors#################
    if 'cursor' in arg:
        a = instr.ask('CURsor?').split(';')
        if 'H' in arg:
            h1 = float(a[10])
            xh1, yh1 = [0, 10000], [((0+Zpos)*Vdiv)+h1, ((0+Zpos)*Vdiv)+h1]
            h2 = float(a[11])
            xh2, yh2 = [0, 10000], [((0 + Zpos) * Vdiv)+h2, ((0 + Zpos) * Vdiv)+h2]
            hd = float(a[12])
            plt.plot(xh1, yh1, c='darkblue', linewidth=2.5)
            plt.plot(xh2, yh2, c='darkblue', linewidth=2.5)
            plt.text(2500, ((0+Zpos)*Vdiv)+h1+(Vdiv*0.2), 'H1', c='darkblue')
            plt.text(2500, ((0+Zpos)*Vdiv)+h2+(Vdiv*0.2), 'H2', c='darkblue')
            plt.scatter(5000, 0, c='black', marker='o', s=1, label='H1 (V) : ' + str(float(a[10])))
            plt.scatter(5000, 0, c='black', marker='o', s=1, label='H2 (V) : ' + str(float(a[11])))
            plt.scatter(5000, 0, c='black', marker='o', s=1, label='ΔH (V) : ' + str(hd))
        if 'V' in arg:
            v1i = float(Hpos) + (float(a[3]) * 1000 / Tbase)
            xv1, yv1 = [v1i, v1i], [(-4*Vdiv), (4*Vdiv)]
            v2i = float(Hpos) + (float(a[4]) * 1000 / Tbase)
            xv2, yv2 = [v2i, v2i], [(-4*Vdiv), (4*Vdiv)]
            vd = float(a[5])
            vdt = float(a[6])
            plt.plot(xv1, yv1, c='darkblue', linewidth=2.5)
            plt.plot(xv2, yv2, c='darkblue', linewidth=2.5)
            plt.text(v1i+200, (2*Vdiv), 'V1', c='darkblue')
            plt.text(v2i+200, (2 * Vdiv), 'V2', c='darkblue')
            plt.scatter(5000, 0, c='black', marker='o', s=1, label='V1 (V) : ' + str(float(a[8])))
            plt.scatter(5000, 0, c='black', marker='o', s=1, label='V2 (V) : ' + str(float(a[9])))
            plt.scatter(5000, 0, c='black', marker='o', s=1, label='ΔV (V) : ' + str(vd))
            plt.scatter(5000, 0, c='black', marker='o', s=1, label='ΔV (s) : ' + str(vdt))
        plt.legend()
    ###############design oscillo############
    x1, y1 = [0, 10000], [0, 0]
    x2, y2 = [5000, 5000], [(-4*Vdiv), (4*Vdiv)]
    plt.plot(x1, y1, c = 'black', linewidth=2)
    plt.plot(x2, y2, c = 'black', linewidth=2)
    plt.scatter(0, ((0+Zpos)*Vdiv), c = 'red', marker = '>', s = 500) # V 0 pos
    plt.scatter(Hpos, (4 * Vdiv), c='red', marker='v', s=500) # H 0 pos
    #############design oscillo2###############
    plt.xticks([0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000])
    plt.yticks(np.arange(-Vdiv * 4, Vdiv * 5, step=Vdiv))
    plt.gca().axes.xaxis.set_ticklabels([])
    plt.gca().axes.yaxis.set_ticklabels([])
    plt.ylim(-Vdiv * 4, Vdiv * 4)
    plt.xlim(0, 10000)
    plt.grid(True)
    plt.xlabel(str(Tbase) + ' s/Div')
    plt.ylabel(str(Vdiv) + ' V/Div')
    plt.title('Selected channel : ' + channel, loc='left')
    plt.tight_layout()
    #ax = plt.axes()               # Uncomment both to change the background color
    #ax.set_facecolor('silver')    # Uncomment both to change the background color
    ###############plot##############
    plt.plot(x, y, c = 'cornflowerblue')

    instr.close()
    return plt

def TDSwfm_live(ip, channel, arg, trace):
    """Same as *TDSwfm()* but it creates a live feed with a refresh rate of ~1.15 sec (0.8670 fps).

    :param ip: IP Address of the oscilloscope
    :type ip: str
    :param channel: Channel selection (*CH1*, *CH2*, *CH3* or *CH4*)
    :type channel: str
    :param arg: *qm* for quick measures, or/and *cursorH* for horizontal cursors, or/and *cursorV* for vertical cursors, or/and *cursorHV* for both type of cursors or *None*
    :type arg: str
    :param trace: *1* or *0* to save a *.txt* of the waveform. *.txt* can be used later with *TDSwfm_hist()*
    :type trace: str
    :Example:
        .. code-block:: python

            TDSwfm_live('192.168.x.x', 'CH1', 'cursorHV', '0')

        *will show the live feed*

        .. image:: ../images/TDS_3x_live.gif
            :align: center

    .. note:: Can only show one channel at a time. Requires NumPy and OpenCV
    .. warning:: Do **not** use arg 1* in *trace*
    """
    while True:
        TDSwfm(ip, channel, arg, trace).savefig('TektroPyx_live.png')
        img = cv2.imread("TektroPyx_live.png")
        cv2.imshow("TektroPyx", img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    cv2.destroyAllWindows()

def TDSwfm_comp(cap1, cap2):
    """Show the comparison between two screenshots taken with TDSwfm().savefig(). It also saves the result as "TektroPyx_comp.jpg" in current path.

    :param cap1: Main image
    :type cap1: str
    :param cap2: Image to compare
    :type cap2: str
    :return: An image with the waveform from cap2 drawn on cap1
    :Example:
        .. code-block:: python

            TDSwfm_comp('image_no1.png', 'image_no2.png')

    *will show the comparison between image_no1 & image_no2 and save the result as "TektroPyx_comp.jpg"* (why jpg ? no one fucking knows)

    .. image:: ../images/TDS_3x_comp.jpg
        :align: center

    .. note:: Requires NumPy and OpenCV. Be careful with the vertical scale and the time base.
    .. warning:: Changing colors will break this function, as it uses mask to detect a given color.
    """
    ###############
    lower = np.array([99, 137, 197]) # If main color (waveform color) has been changed, lower & upper has to be changed
    upper = np.array([119, 157, 277])# in accordance with chosen color
    ###############

    frame = cv2.imread(cap1, cv2.IMREAD_UNCHANGED)
    comp = cv2.imread(cap2, cv2.IMREAD_UNCHANGED)

    hsv = cv2.cvtColor(comp, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)

    _, threshold = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        cv2.drawContours(frame, [cnt], -1, (255, 220, 120), -1)

    cv2.imshow('frame', frame)
    cv2.imwrite('TektroPyx_comp.jpg', frame)
    cv2.waitKey(0)

def TDSwfm_hist(ntrace):
    """Return the waveform saved in the *trace_xxx.txt* given in argument

    :param ntrace: title of .txt file
    :type ntrace: str
    :return: *waveform*
    :rtype: *pyplot object*
    :Example:
        .. code-block:: python

            TDSwfm_hist('trace_113807443533').show()

        *will show the waveform*

        .. code-block:: python

            TDSwfm_hist('trace_113807443533').savefig('TektroPyx_hist.png')

        *will save the waveform as "TektroPyx_hist.png"*

    .. note:: Please, refer to TDSwfm() as it's basically the same function
    """
    param = open(str(ntrace) + ".txt", "r")
    trace = param.readlines(-1)
    param.close()

    YZE = float((trace[0].replace('\n', '')))
    YMU = float((trace[1].replace('\n', '')))
    YOF = float((trace[2].replace('\n', '')))
    Vdiv = float((trace[3].replace('\n', '')))
    Tbase = float((trace[4].replace('\n', '')))
    Zpos = float((trace[5].replace('\n', '')))
    Hpos = float((trace[6].replace('\n', '')))
    wfm_data = trace[7].replace('\n', '')
    #####################################
    wfm_data = wfm_data.split(',')

    y = []
    x = []
    i = 1
    plt.clf()

    for point in wfm_data:
        y.append((YZE + (YMU * (int(point) - YOF))) + (Zpos * Vdiv))
        x.append(i)
        i += 1

    ###############design oscillo############
    x1, y1 = [0, 10000], [0, 0]
    x2, y2 = [5000, 5000], [(-4 * Vdiv), (4 * Vdiv)]
    plt.plot(x1, y1, c='black', linewidth=2)
    plt.plot(x2, y2, c='black', linewidth=2)
    plt.scatter(0, ((0 + Zpos) * Vdiv), c='red', marker='>', s=500)  # V 0 pos
    plt.scatter(Hpos, (4 * Vdiv), c='red', marker='v', s=500)  # H 0 pos
    #############design oscillo2###############
    plt.xticks([0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000])
    plt.yticks(np.arange(-Vdiv * 4, Vdiv * 5, step=Vdiv))
    plt.gca().axes.xaxis.set_ticklabels([])
    plt.gca().axes.yaxis.set_ticklabels([])
    plt.ylim(-Vdiv * 4, Vdiv * 4)
    plt.xlim(0, 10000)
    plt.grid(True)
    plt.xlabel(str(Tbase) + ' s/Div')
    plt.ylabel(str(Vdiv) + ' V/Div')
    plt.title('trace N°'+str(ntrace), loc='left')
    plt.tight_layout()
    # ax = plt.axes()               # Uncomment both to change the background color
    # ax.set_facecolor('silver')    # Uncomment both to change the background color
    ###############plot##############
    plt.plot(x, y, c='cornflowerblue')

    return plt

def TDSwfm_meas(ip, mes):
    """Return the wanted measure

    :param ip: IP Address of the oscilloscope
    :type ip: str
    :param mes: Measure to return, see note for more informations
    :type mes: str
    :return: *measure*
    :rtype: str
    
    .. note:: Available measures : AMPlitude, AREa, BURst, CARea, CMEan, CRMs, DELAY, FALL,
        FREQuency, HIGH, LOW, MAXimum, MEAN, MINImum, NDUty, NOVershoot, NWIdth, PDUty,
        PERIod, PHASE, PK2pk, POVershoot, PWIdth, RISe, RMS
    """
    #########
    #AMPlitude | AREa | BURst | CARea | CMEan |
    #CRMs | DELAY | FALL | FREQuency | HIGH | LOW |
    #MAXimum | MEAN | MINImum | NDUty | NOVershoot |
    #NWIdth | PDUty | PERIod | PHASE | PK2pk | POVershoot |
    #PWIdth | RISe | RMS
    #########
    try:
        instr =  vxi11.Instrument(ip)
        EOL = '\n'
        instr.write('measurement:immed:type ' + mes)
        a = instr.ask('measurement:immed:value?'+EOL)
        instr.close()
        return a
    except:
        return 'error'

def TDSwfm_trig(ip):
    """Return the trigger state

    :param ip: IP Address of the oscilloscope
    :type ip: str
    :return: *trigger* state
    :rtype: str
    """
    try:
        instr =  vxi11.Instrument(ip)
        EOL = '\n'
        a = instr.ask('trigger:state?'+EOL)
        instr.close()
        return a
    except:
        return 'error'

def CMD_VXI(ip, cmd):
    """To test any command using VXI-11 protocol

        :param ip: IP Address
        :type ip: str
        :param cmd: Command to send
        :type cmd: str
        :return: *response* if query, otherwise *None*
        :rtype: str

        .. warning:: *EOL* is a *line feed* by default and **can only be changed in source**
        """
    try:
        instr =  vxi11.Instrument(ip)
        EOL = '\n'
        if '?' in cmd:
            a = instr.ask(cmd+EOL)
        else:
            a = instr.write(cmd+EOL)
        instr.close()
        return a
    except:
        return 'error'

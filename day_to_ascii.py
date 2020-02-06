import pandas as pd
import os
import datetime

duration = 60
channels = ["STRA.EHE"] #, "STRA.EHN", "STRA.EHZ", ]
day_file_path = "/home/llayer/Data/Stromboli/day/"
day_file_rec_path = "/home/llayer/Data/Recovered/day/"
out_path = "/home/llayer/Data/ascii_test/"
tot_samples = 3050

corr_files = 0

def dayls(date, starttime, endtime, channel):
    
        
    # String for the day file location 
    day_file = day_file_path + date + "/" + channel + "/" + channel.replace(".","_") \
                + "." + date.replace("/", "") + starttime[0:2] + "0000.day"

    if os.path.isfile(day_file) == False:
        day_file = day_file_rec_path + date + "/" + channel + "/" + channel.replace(".","_") \
                + "." + date.replace("/", "") + starttime[0:2] + "0000.day"

    #Output dir
    outfile_path = out_path + date 
    if not os.path.exists(outfile_path):
        os.makedirs(outfile_path)

    outfile = outfile_path + "/" + starttime.replace(":", "") + "_" + endtime.replace(":", "") + "_" + \
                channel.replace(".","_") + ".ascii"

    dayls_command = "dayls -d -l -s " + date + "." + starttime + " -e " + date  + "." + endtime + " " + \
                    day_file + " " + outfile 

    print( dayls_command )


    try:
        os.system(dayls_command)
    except:
        print( "Problems with file:", day_file )

    return outfile
    
    
def slist_header( channel, nsamples, starttime ):
    
    # needed for the conversion to obspy
    
    HEADER = ("TIMESERIES {network}_{station}_{location}_{channel}_{dataquality}, "
              "{npts:d} samples, {sampling_rate} sps, {starttime!s:.26s}, "
              "{format}, {dtype}, {unit}")
    
    network = 'Stromboli'
    station = 'STRA'
    location = 'stromboli'
    dataquality = 'D'
    sampling_rate = 50
    format = 'SLIST'
    dtype = 'INTEGER'
    unit = ''
    
    header = HEADER.format(
    network=network, station=station, location=location,
    channel=channel, dataquality=dataquality, npts=nsamples,
    sampling_rate=sampling_rate, starttime=starttime,
    format=format, dtype=dtype, unit=unit)
    return header
    
            
def to_ascii(time, ms):
        
    date, starttime = time.split()
    date = "20" + date  

    # Calculate the time period
    datetime_start = datetime.datetime.strptime(starttime, '%H:%M:%S')
    datetime_end = (datetime_start + datetime.timedelta(seconds=duration)).strftime("%H:%M:%S")

    endtime = str(datetime_end)
    
    for channel in channels:

        outfile_path = ''
        if int(endtime[0:2]) != int(starttime[0:2]) :

            switch_time = endtime[0:2] + ":00:00"

            first_file = dayls(date, starttime, switch_time, channel)
            second_file = dayls(date, switch_time, endtime, channel)

            outfile_path = out_path + date + "/" + starttime.replace(":", "") + "_" + endtime.replace(":", "") + "_" + \
                           channel.replace(".","_") + ".ascii"

            print(outfile_path)
            
            os.system("cat " + first_file + " " + second_file + " > " + outfile_path)
            os.system("rm " + first_file + " " + second_file)


        else:
            outfile_path = dayls(date, starttime, endtime, channel)
        
        nsamples = sum(1 for i in open(outfile_path, 'rb'))
        
        if nsamples != tot_samples:
            
            global corr_files
            corr_files += 1
        
        header = slist_header( channel.split('.')[1], nsamples, date.replace('/', '-') + 'T' + 
                              starttime + '.' + ms + '000' )
        
        #print( "echo \"" + header + "\" | cat - " + outfile_path + " > temp" ) # && mv temp " + outfile_path )
        os.system("echo \"" + header + "\" | cat - " + outfile_path + " | sponge " + outfile_path )
    
    #print date, time, str(endtime)
    

def read_ctg(filepath):
    
    data = pd.read_csv(filepath, header = None)
    data = data.drop([2],axis=1)
    data = data.drop([0])
    data.columns = ["time", "ms"]
    return data

def test():
    
    cat_path = '/home/llayer/Downloads/FinalCatalogue/'
    filepath = cat_path + 'STRA_LHE.20190515.ctg'
    data = read_ctg( filepath )
    times = [ to_ascii(x,y) for x, y in zip(data['time'], data['ms'])]
    print( 'Corrupted files:', corr_files) 
    
def convert_all():
    
    import glob
    cats = glob.glob(cat_path + "*.ctg")
    for cat in cats:
        data = reat_ctg( cat )
        times = [ to_ascii(x,y) for x, y in zip(data['time'], data['ms'])]
        print( 'Corrupted files:', corr_files) 
    
    
test()
#convert_all()
    
    
    
        
        


import pandas as pd
import os
import datetime

duration = 61
stations = ["STRA.EHN", "STRA.EHZ", "STRA.EHE"]
day_file_path = "/home/llayer/Data/Stromboli/day/"
day_file_rec_path = "/home/llayer/Data/Recovered/day/"
out_path = "/home/llayer/Data/ascii/"
cat_path = '/home/llayer/Downloads/FinalCatalogue/'

def dayls(date, starttime, endtime):
    
    for station in stations:

        # String for the day file location 
        day_file = day_file_path + date + "/" + station + "/" + station.replace(".","_") \
                    + "." + date.replace("/", "") + starttime[0:2] + "0000.day"

        if os.path.isfile(day_file) == False:
            day_file = day_file_rec_path + date + "/" + station + "/" + station.replace(".","_") \
                    + "." + date.replace("/", "") + starttime[0:2] + "0000.day"

        #Output dir
        outfile_path = out_path + date 
        if not os.path.exists(outfile_path):
            os.makedirs(outfile_path)

        outfile = outfile_path + "/" + starttime.replace(":", "") + "_" + str(endtime).replace(":", "") + "_" + \
                    station.replace(".","_") + ".ascii"

        dayls_command = "dayls -d -s " + date + "." + starttime + " -e " + date  + "." + str(endtime) + " " + \
                        day_file + " " + outfile 
        print dayls_command

        try:
            os.system(dayls_command)
        except:
            print
            print "Problems with file:"
            print day_file
            print
            
            
def to_ascii(data):

    for row in data:

        date, time = row.split()
        date = "20" + date

        # Calculate the time period
        starttime = datetime.datetime.strptime(time, '%H:%M:%S')
        endtime = (starttime + datetime.timedelta(seconds=duration)).strftime("%H:%M:%S")

        if int(time[3:5]) > 59:
            continue
    
        dayls(date, time, endtime)

        #print 
        #print date, time
        
        
def test():
    
    filepath = cat_path + 'STRA_LHE.20190515.ctg'
    data = pd.read_csv(filepath, header = None)
    data = data[0]
    data = data.drop_duplicates()
    sample = data.iloc[0:2]
    to_ascii(sample)
    
def convert_all():
    
    import glob
    cats = glob.glob(cat_path + "*.ctg")
    for cat in cats:
    data = pd.read_csv(cat, header = None)
    data = data[0]
    data = data.drop_duplicates()
    to_ascii(data)
    
    
test()
    
    
    
        
        


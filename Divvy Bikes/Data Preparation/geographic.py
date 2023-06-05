from geopy.geocoders import Nominatim
import pandas as pd
import numpy as np
import math

#Function to add geographic data by chunk
def addGeographicChunk(df, lat, long, by_chunk = False, chunk_size = 10000):
    """
    Function to add geographic data to a dataframe containing coordinate values
    @params
        - df: pandas dataframe
        - lat: list or array of latitude values 
        - long: list or array of longtitude values
        - by_chunk: boolean
        - chunk_size: int
    """
    nrow = len(df)
    final_df = pd.DataFrame()
    if by_chunk:
        chunks = np.array_split(df, math.ceil(nrow / chunk_size))
        lats = np.array_split(lat, math.ceil(nrow / chunk_size))
        lons = np.array_split(long, math.ceil(nrow / chunk_size))
        for i in range(len(chunks)):
            curr_chunk = chunks[i]
            curr_lat = lats[i]
            curr_long = lons[i]
            #Create lat/long input
            zipped = list(zip(curr_lat, curr_long))
            lat_long_input = [str(l1) + ", " + str(l2) for l1, l2 in zipped ]
            
            #geolocator object
            geolocator = Nominatim(user_agent= "divvy-bikes")

            #Loop to retrive demographic data
            neighbourhood = []
            suburb = []
            city = []
            county = []
            zip_code = []
            for coords in lat_long_input:
                #Retrieve location information
                location = geolocator.reverse(coords)
                location = location.raw
                location = location.get("address")

                #Populate column lists
                neighbourhood.append(location.get("neighbourhood"))
                suburb.append(location.get("suburb"))
                city.append(location.get("city"))
                county.append(location.get("county"))
                zip_code.append(location.get("postcode"))

            #Add geographic columns to dataframe
            curr_chunk = curr_chunk.assign(neighbourhood = neighbourhood,
                        suburb = suburb,
                        city = city,
                        county = county,
                        zip_code = zip_code)
            
            final_df = pd.concat([final_df, curr_chunk])
            


    return final_df



#Function to add geographic data
def addGeographic(df, lat, long):
    """
    Function to add geographic data to a dataframe containing coordinate values
    @params
        - df: pandas dataframe
        - lat: list or array of latitude values 
        - long: list or array of longtitude values
    """
    
    #Create lat/long input
    zipped = list(zip(lat, long))
    lat_long_input = [str(l1) + ", " + str(l2) for l1, l2 in zipped ]
    
    #geolocator object
    geolocator = Nominatim(user_agent= "divvy-bikes")

    #Loop to retrive demographic data
    neighbourhood = []
    suburb = []
    city = []
    county = []
    zip_code = []
    for coords in lat_long_input:
        #Retrieve location information
        location = geolocator.reverse(coords)
        location = location.raw
        location = location.get("address")

        #Populate column lists
        neighbourhood.append(location.get("neighbourhood"))
        suburb.append(location.get("suburb"))
        city.append(location.get("city"))
        county.append(location.get("county"))
        zip_code.append(location.get("postcode"))

    #Add geographic columns to dataframe
    df = df.assign(neighbourhood = neighbourhood,
                   suburb = suburb,
                   city = city,
                   county = county,
                   zip_code = zip_code)

    return df



#Function to add zip code information
def addZip(df, lat, long):
    """
    Function to add zip code data to a dataframe containing coordinate values
    @params
        - df: pandas dataframe
        - lat: list or array of latitude values 
        - long: list or array of longtitude values
    """
    
    #Create lat/long input
    zipped = list(zip(lat, long))
    lat_long_input = [str(l1) + ", " + str(l2) for l1, l2 in zipped ]
    
    #geolocator object
    geolocator = Nominatim(user_agent= "divvy-bikes")

    #Loop to retrive zip code data
    zip_code = []
    for coords in lat_long_input:
        #Retrieve location information
        location = geolocator.reverse(coords)
        location = location.raw
        location = location.get("address")

        #Populate zip code column
        zip_code.append(location.get("postcode"))

    #Add geographic columns to dataframe
    df = df.assign(zip_code = zip_code)

    return df



#Function to impute missing zip code values for stations data
def imputeStationsZip(stations, zip_column_index):
    """
    Function to impute missing zip code data to the stations dataframe
    @params
        - stations: Divvy Bikes stations dataframe
    """

    #Manually impute missing values
    stations.iloc[43, zip_column_index] = "60654"
    stations.iloc[46, zip_column_index] = "60605"
    stations.iloc[370, zip_column_index] = "60609"

    return stations



#Funciton to impute missing zip code values for landmarks data
def imputeLandmarksZip(landmarks):
    """
    Function to impute missing zip code data to the landmarks dataframe
    @params
        - landmarks: landmarks dataframe
    """

    #Manually impute missing value
    landmarks.iloc[131,9] = "60611"

    return landmarks



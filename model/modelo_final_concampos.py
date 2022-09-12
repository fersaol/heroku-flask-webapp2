import sys
import os
from typing import Union
from pathlib import Path
sys.path.append(os.getcwd() + "\\utils")

import pandas as pd
import numpy as np
from utils import general_purpose as gp
import joblib
from sklearn.preprocessing import PowerTransformer,RobustScaler
import datetime
class Final_Model:

    """Clase que trata los datos y realiza la clusterización, predicción y 
    clasificación, convina los tres resultados y devuelve una explicación
    
    ----------------------------------------------------
    # Args:
        Country: (str) country which are refering the forecasts
        - Year: (str) aaaa-mm-dd format
        - GDP: (float | int) previsional or real gdp
        - Population: (float | int) actual population
        - Energy_production: (float | int) energy production till now or previsional
        - Energy_consumption: (float | int) energy consumption till now or previsional
        - CO2_emission: (float | int) co2 emitted till now or previsional
        - energy_type: (int) one of {0:renewables,1:nuclear,2:gas,3:petroleum and other liquids,4:coal}
    
    ------------------------------------------------------
    # Return
        if using the run_whole_model() method it runs all the functions and
        returns a str statement about the classification and the efficiency
        forecasted."""

    def __init__(self,Country:str,Year:str,GDP:Union[int,float],
                Population:Union[int,float],Energy_production:Union[int,float],
                Energy_consumption:Union[int,float],CO2_emission:Union[int,float],
                energy_type:Union[int,float]):
        self.ruta = Path.cwd()/"model"
        self.df = gp.dataframes_charger("df_clusters_v1.csv")
        self.Year = Year
        self.Country = Country.capitalize()
        self.GDP = GDP
        self.Population = Population
        self.Energy_production = Energy_production
        self.Energy_consumption = Energy_consumption
        self.CO2_emission = CO2_emission
        self.energy_type = energy_type
        self.per_capita_production = self.Energy_production/self.Population
        self.Energy_intensity_by_GDP = self.Energy_production/self.GDP
        self.balance = self.Energy_production - self.Energy_consumption
        self.energy_dependecy = self.Energy_consumption/self.GDP
        self.use_intensity_pc = self.Energy_consumption/self.Population
        self.co2_pc = self.CO2_emission/self.Population
        self.df_preproc = Final_Model.preprocessing(self,PowerTransformer())
        self.eficiency = Final_Model.regression(self)[0]
        self.latitude = Final_Model.coordinates(self,self.Country)[0]
        self.longitude = Final_Model.coordinates(self,self.Country)[1]
        self.Energy_type = Final_Model.energy_source(self,self.energy_type)
        self.CODE_x = Final_Model.internacional_code(self,self.Country)
        self.continent = Final_Model.selec_continent(self,self.Country)
        self.clusters = Final_Model.clustering(self)[0]
        
        
        
    def energy_source(self,code: int) -> str:

        """it takes the energy type code provided when constructing the object and
        returns the energy type name related with it.
        
        --------------------------------------------------------
        # Args:
            code: (int) one of {0:renewables,1:nuclear,2:gas,3:petroleum and other liquids,4:coal}
        
        ---------------------------------------------------------
        # Returns:
            a string with the name related with the code provided."""

        df = self.df
        e_types = dict(zip(df.energy_type.unique(),df.Energy_type.unique()))
        return e_types[code]
        
    def selec_continent(self,country: str) -> str:

        """It selects the continent related to the country

        -----------------------------------------------------------
        # Args:
            country: (str) the country which data belongs to
            
        -----------------------------------------------------------
        # Returns:
            a string with the continent related """

        df = self.df
        count_groups = df.groupby("continent")["Country"]
        dic_continent = {con:np.unique(coun.values) for con,coun in count_groups}

        for cont in dic_continent.keys():
            if country in dic_continent[cont]:
                return cont

    def internacional_code(self,country: str) -> str:

        """It provides the country's international code for map representation
        
        ----------------------------------------------------------
        # Args:
            country: (str) the country we want the international code of
        
        ----------------------------------------------------------
        # Returns:
            a string with the ISO code for the country provided"""

        df = self.df
        dic_code = {coun:df.CODE_x.unique()[cod] for cod,
                    coun in enumerate(df.Country.unique())}
        return dic_code[country]

    def coordinates(self,country: str) -> tuple:
        """It provides the latitude and logitude related with the country
        
        ---------------------------------------------------------
        # Args:
            country: (str)
            
        ---------------------------------------------------------
        # Returns:
            s tuple with the country's coordinates, latitude and longitude"""

        df = self.df
        lat_lon = df.groupby("Country")[["latitude","longitude"]].mean()
        dic_coors = {count:lat_lon.loc[count].values for count in lat_lon.index}
        return dic_coors[country]

    def registration(self):

        """Updates the source dataframe with the new data provided regardless
        these are previsional or definitive
        
        ----------------------------------------------------------
        # Args:
            no args
            
        ----------------------------------------------------------
        # Returns:
            updates the source dataframe by replacing it with a new one"""

        destino = Path.cwd()/"data/processed/"

        df = self.df
        df.loc[len(df)] = np.array([self.GDP, self.Population,
                                self.Energy_production,self.Energy_consumption,
                                self.CO2_emission,self.per_capita_production, 
                                self.Energy_intensity_by_GDP,self.balance, 
                                self.eficiency, self.energy_dependecy,
                                self.use_intensity_pc,self.co2_pc, self.latitude, 
                                self.longitude, self.Year, self.Country, 
                                self.Energy_type,self.CODE_x, self.continent,
                                self.clusters, self.energy_type])


        df.to_pickle(destino/"updated_data.pkl")

    def preprocessing(self,escalado) -> pd.DataFrame:

        """It preprocesses the new data provided so it is usable by the models
        
        ----------------------------------------------------------
        # Args:
            escalado: (sklearn.preprocesing) a valid scaler form sklearn
            
        # Returns:
            the new data in a pd.Dataframe with column names"""

        data_df = self.df
        data_df = data_df.select_dtypes(exclude="object")
        not_scale = ["latitude","longitude","clusters","energy_type","eficiency"]
        nd_columns = [x for x in data_df.columns if x not in not_scale]

        data_fit = data_df.loc[:,nd_columns]
        working_data = np.array([self.GDP,self.Population,self.Energy_production,
                                self.Energy_consumption,self.CO2_emission,
                                self.per_capita_production,self.Energy_intensity_by_GDP,
                                self.balance,self.energy_dependecy,self.use_intensity_pc,
                                self.co2_pc])

        new_data = pd.DataFrame(working_data.reshape(1,-1),columns=nd_columns)

        scaler = escalado.fit(data_fit)
        
        return pd.DataFrame(scaler.transform(new_data),
                            columns=scaler.get_feature_names_out())


    def clustering(self) -> int:

        """It returns the cluster the data belongs to
        
        ---------------------------------------------------------
        # Args:
            no args
            
        ---------------------------------------------------------
        # Returns:
            prediction as integer wich is one of the 4 clusters availables."""

        # 1. Preprocesado y selección de variables
        df = self.df_preproc
        clus_df = df[["CO2_emission","Energy_production"]]

        # 2. carga del modelo
        clustering = joblib.load(self.ruta/"ClusteringModel.pkl")
        # 3. devuelve la predicción
        pred = clustering.predict(clus_df)
        return pred

    def regression(self) -> float:
        
        """It returns the efficiency value for the data provided
        
        ---------------------------------------------------------
        # Args:
            no args
            
        ---------------------------------------------------------
        # Returns:
            prediction as float."""

        cluster = Final_Model.clustering(self)
        reg_vars = {
            0:['balance', 'Energy_consumption', 'Energy_production', 'CO2_emission'],
            1:['CO2_emission', 'co2_pc', 'per_capita_production', 'Energy_consumption'],
            2:['GDP', 'Population', 'Energy_consumption', 'CO2_emission', 'balance'],
            3:['CO2_emission', 'Energy_production', 'balance', 'Energy_consumption']
                    }

        if cluster == 0:
            df = self.df_preproc[reg_vars[0]]
            clus_df = df.rename(columns=dict(zip(df.columns,df.columns.str.lower())))

            reg_0 = joblib.load(self.ruta/"reg_cluster0.pkl")
            pred = reg_0.predict(clus_df)
        elif cluster == 1:
            df = Final_Model.preprocessing(self,escalado=RobustScaler())
            df = self.df_preproc[reg_vars[1]]
            clus_df = df.rename(columns=dict(zip(df.columns,df.columns.str.lower())))

            reg_1 = joblib.load(self.ruta/"reg_cluster1.pkl")
            pred = reg_1.predict(clus_df)
        elif cluster == 2:
            df = self.df_preproc[reg_vars[2]]
            clus_df = df.rename(columns=dict(zip(df.columns,df.columns.str.lower())))

            reg_2 = joblib.load(self.ruta/"reg_cluster2.pkl")
            pred = reg_2.predict(clus_df)
        else:
            df = self.df_preproc[reg_vars[3]]
            clus_df = df.rename(columns=dict(zip(df.columns,
                                                    df.columns.str.lower())))
            
            reg_3 = joblib.load(self.ruta/"reg_cluster3.pkl")
            pred = reg_3.predict(clus_df)

        return pred

    def classification(self) -> int:
        
        """It returns the class the data belongs to
        
        ---------------------------------------------------------
        # Args:
            no args
            
        ---------------------------------------------------------
        # Returns:
            prediction as integer wich is one of the 4 classes a country can
            belongs to."""

        # 1. selección de variables
        vars_rf = ['GDP', 'Population', 'Energy_consumption',
                    'per_capita_production','Energy_intensity_by_GDP', 'balance',
                    'energy_dependecy','co2_pc']
        df = self.df_preproc
        clas_df = df[vars_rf]
        clas_df = clas_df.rename(columns=dict(zip(clas_df.columns,
                                                clas_df.columns.str.lower())))
        clas_df["energy_type"] = self.energy_type # la metemos aparte porque no debe ser preprocesada

        # 2. carga del modelo
        model_class = joblib.load(self.ruta/"RanFor_Classifier.pkl")
        # 3. devuelve la predicción
        pred = model_class.predict(clas_df)
        return pred

    def run_whole_model(self) -> str:
        
        """Runs all the needed functions for the whole model and provides with
        a clear explanaition and recomendations of what means belonging to that
        cluster.
        
        ---------------------------------------------------------
        # Args:
            no args
            
        ---------------------------------------------------------
        # Returns:
            a string with the cluster the country belongs to, its efficiency and
            a explanation of what means being in that cluster as well as a
            recommendation."""

        tag = Final_Model.classification(self)[0]
        efi = round(Final_Model.regression(self)[0],3)
        Final_Model.registration(self)

        if tag == 0:
            return f"""
                    -----------------------------------------------------------
                    The efficiency predicted for your country is {efi},
                    what means it is classified in the environmental group {tag}.
                    This group is characterized by the following description:\n

                    ### LOW PRODUCTION-HIGH CONTAMINATION
                    The energy production is low but it is not the lower compared
                    with the rest of the world energy producers. The production
                    is based on natural gas, petroleum and coal and because of
                    this energy mix the co2 emissions are high.

                    ### RECOMMENDATION
                    Your efficiency can improve a lot since your energy production
                    mix is not optimal. Focus on changing your energy sources.

                    -----------------------------------------------------------
                    """

        elif tag == 1:
            return f"""
                    -----------------------------------------------------------
                    The efficiency predicted for your country is {efi}, 
                    what means it is classified in the environmental group {tag}.
                    This group is characterized by the following description:

                    ### LOW PRODUCTION-LOW CONTAMINATION
                    The energy production is low but the contamination it also is.
                    In this group the production comes mainly from petroleum but
                    also from renewables and natural gas. The energy mix is not ideal,
                    but the emitted co2 has no great impact on environment

                    ### RECOMMENDATION
                    As the production remains steady the country can continues this
                    way. But if the aim is to increase energy production the mix
                    should be improved in order to lower the co2 emissions. Reinforce
                    renewables.
                    
                    -----------------------------------------------------------
                    """

        elif tag == 2:
            return f"""
                    
                    The efficiency predicted for your country is {efi}, 
                    what means it is classified in the environmental group {tag}.
                    This group is characterized by the following description:\n

                    VERY HIGH PRODUCTION-VERY HIGH CONTAMINATION\n
                    The energy production is very high and contamination too, so
                    you are one of the world's major suppliers. The production in
                    this group comes normally from petroleum, coal and natural
                    gas.\n

                    RECOMMENDATION\n
                    Your country has great impact on environmental care so it would
                    be good diversify the production mix enhancing renewables and
                    natural gas if possible. In any case, reducing coal and pretroleum
                    would be great.
                    """

        else:
            return f"""
                    -----------------------------------------------------------
                    The efficiency predicted for your country is {efi}, 
                    what meansit is classified in the environmental group {tag}.
                    This group is characterized by the following description:

                    ### GOOD BALANCE BETWEEN PRODUCTION AND CONTAMINATION
                    The production amount is good, coming from a good balanced production
                    mix and using all of them proportionally.

                    ### RECOMMENDATION
                    Just keep this way, your country is environmental friendly and
                    knows how to balance production and world care.

                    -----------------------------------------------------------
                    """ 


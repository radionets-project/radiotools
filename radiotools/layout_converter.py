import pandas as pd
import numpy as np
from pathlib import Path

class Layout:
    def __init__(self):
        None
    
    def save(self, path, fmt="pyvisgen", overwrite=False):
        
        """
        Saves the layout to a layout file.

        Parameters
        ----------
        path : str
            The path of the file to save the array layout to.

        fmt : str, optional
            The layout format the output file is supposed to have (available: casa, pyvisgen) (default is pyvisgen).

        overwrite : bool, optional
            Whether to overwrite the file if it already exists (default is False).
        """
        
        FORMATS = ["casa", "pyvisgen"]
        
        file = Path(path)
        
        if file.exists():
            if overwrite:
                file.unlink()
            else:
                raise FileExistsError(f"The file {file} already exists! If you want to override it set overwrite=True!")
                

        data = []
        
        match fmt:
            case "pyvisgen":
                data.append("station_name X Y Z dish_dia el_low el_high SEFD altitude\n")

                for i in range(0, len(self.x)):
                    row = map(str,
                              [self.names[i], self.x[i], self.y[i], self.z[i], 
                               self.dish_dia[i], self.el_low[i], self.el_high[i], self.sefd[i], self.altitude[i]])
                    data.append("\t".join(row) + "\n")
            
            case "casa":
                data.append("# X Y Z dish_dia station_name\n")
                
                for i in range(0, len(self.x)):
                    row = map(str,[self.x[i], self.y[i], self.z[i], 
                                   self.dish_dia[i], self.names[i]])
                    print(row)
                    data.append(" ".join(row) + "\n")
                
            case _:
                raise ValueError(f"{fmt} is not a valid format! Possible formats are: {', '.join(FORMATS)}!")
                
        
        

        with open(file, 'w', encoding='utf-8') as f: 
            f.writelines(data)
    
    @classmethod
    def from_casa(cls, cfg_path, el_low=15, el_high=85, sefd=0, altitude=0):
        """
        Import a layout from a NRAO CASA layout config.

        Parameters
        ----------
        cfg_path : str
            The path of the config file to import.

        el_low : float or array_like, optional
            The minimal elevation in degrees the telescope can be adjusted to.
            If provided as singular number all telescopes in the array will be assigned the same value.
            
        el_high : float or array_like, optional
            The maximal elevation in degrees the telescope can be adjusted to.
            If provided as singular number all telescopes in the array will be assigned the same value.

        sefd : float or array_like, optional
            The system equivalent flux density of the telescope.
            If provided as singular number all telescopes in the array will be assigned the same value.
            
        altitude : float or array_like, optional
            The altitude of the telescope.
            If provided as singular number all telescopes in the array will be assigned the same value.
        """
        
        df = pd.read_csv(cfg_path, delimiter="\s+", encoding="utf-8", skip_blank_lines=True, skiprows=1,
                        names=["x", "y", "z", "dish_dia", "station_name"],
                        dtype={'x': float, 'y': float, "z": float, "dish_dia": float, 'station_name': str}, comment='#')
        cls = cls()
        cls.x = df.iloc[:,0].to_list()
        cls.y = df.iloc[:,1].to_list()
        cls.z = df.iloc[:,2].to_list()
        cls.dish_dia = df.iloc[:,3].to_list()
        cls.names = df.iloc[:,4].to_list()
        cls.el_low = np.repeat(el_low, len(cls.x))
        cls.el_high = np.repeat(el_high, len(cls.x))
        cls.sefd = np.repeat(sefd, len(cls.x))
        cls.altitude = np.repeat(altitude, len(cls.x))
        
        return cls
    
    
    @classmethod
    def from_pyvisgen(cls, cfg_path):
        """
        Import a layout from a NRAO CASA layout config.

        Parameters
        ----------
        cfg_path : str
            The path of the config file to import.
        """
        
        df = pd.read_csv(cfg_path, delimiter="\s+", encoding="utf-8", skip_blank_lines=True, skiprows=1,
                        dtype={'station_name': str, 'x': float, 'y': float, "z": float, "dish_dia": float, 
                               "el_low": float, "el_high": float, "sefd": float, "altitude": float})
        cls = cls()
        cls.names = df.iloc[:,0].to_list()
        cls.x = df.iloc[:,1].to_list()
        cls.y = df.iloc[:,2].to_list()
        cls.z = df.iloc[:,3].to_list()
        cls.dish_dia = df.iloc[:,4].to_list()
        cls.el_low = df.iloc[:,5].to_list()
        cls.el_high = df.iloc[:,6].to_list()
        cls.sefd = df.iloc[:,7].to_list()
        cls.altitude = df.iloc[:,8].to_list()
                
        return cls
        
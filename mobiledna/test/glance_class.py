# Processing glance files

from mobiledna.core.appevents import Appevents
from os.path import join,pardir
from os import listdir
from mobiledna.core import help as hlp
from tqdm import tqdm
import numpy as np
if __name__ == '__main__':

    hlp.hi()
    hlp.set_param(log_level=1)
    hlp.set_dir(join(pardir, pardir, 'data', 'glance','objects','appevents'))

    # Set dirs
    ae_objects_dir = join(pardir, pardir, 'data', 'glance','objects','appevents')
    ae_dir = join(pardir, pardir, 'data','glance','appevents')
    ae_processed_dir = join(pardir, pardir, 'data','glance','processed_appevents')

    # Get file names
    ae_data_files = sorted(listdir(ae_dir))
    log_periods = {}

    # Loop over files and process
    for ae_data_file in tqdm(ae_data_files, desc='Processing appevents'):

        # File name
        file_name = ae_data_file.split('.')[0]

        # Set paths
        ae_path = join(ae_dir, ae_data_file)
        ae_object_path = join(ae_objects_dir, f'{file_name}.ae')

        # Load
        ae_object = Appevents.load_data(path=ae_path)
        ae_object.strip(number_of_days=120)

        # Get number of log days
        try:
            log_periods[file_name] = ae_object.get_dates()[0]

            if len(log_periods[file_name]) == 120:
                ae_object.save_data(dir=ae_processed_dir,name=file_name)
        except Exception as e:
            print(f'Failed for {file_name}: {e}.')

    # Get good loggers (more than 120 days of logging)
    good_loggers = {id: val for id, val in log_periods.items() if len(val) >= 120}

    np.save(join(pardir, pardir, 'data', 'good_loggers.npy'), good_loggers)

        # Process
        #ae_object.add_category(scrape=True)
        #ae_object.add_date_annotation()
        #ae_object.strip()

        # Save
        #ae_object.to_pickle(path=ae_object_path)
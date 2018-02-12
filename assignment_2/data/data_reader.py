# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
:Date: 2018. 2. 9.
"""
from os import walk
from datetime import datetime

import pandas as pd

from assignment_2.data.president import all_president_names

SPEECH_DIR = 'assignment_2/data/speech/'


def get_speeches(selected_presidents=None):
    """

    :param selected_presidents: (list[str]) The list of president names.

    :return speeches: (DataFrame)
        index   date        | (datetime)
                president   | (str)
        columns script      | (str)
    """
    if selected_presidents is None:
        selected_presidents = all_president_names

    all_file_names = []
    for (dirpath, dirnames, filenames) in walk(SPEECH_DIR):
        all_file_names.extend(filenames)

    # Over iterating all_file_names, save date, president, script.
    dates = []
    presidents = []
    scripts = []
    for file_name in all_file_names:
        with open(SPEECH_DIR + file_name, 'r', encoding='unicode_escape') as file:
            # file_name example
            # 2010-01-01 president_name.txt
            date = datetime.strptime(file_name[:10], '%Y-%m-%d')
            president = file_name[11:-4]
            script = file.read().replace('\n', '')
            if president in selected_presidents:
                dates.append(date)
                presidents.append(president)
                scripts.append(script)

    # Make speeches dataframe by dates, presidents, scripts.
    speeches = pd.DataFrame(data={
        'date': dates,
        'president': presidents,
        'script': scripts,
    })

    # Set date and president for index.
    speeches = speeches.set_index(['date', 'president'])

    return speeches


if __name__ == '__main__':
    speeches = get_speeches()
    print(len(speeches))
    print(speeches.head())

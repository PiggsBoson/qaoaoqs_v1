import seaborn as sns
import pandas as pd
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
import json
import os
import re
import numpy as np


"""
Using the plotter:

Call it from the command line, and supply it with logdirs to experiments.
Suppose you ran an experiment with name 'test', and you ran 'test' for 10 
random seeds. The runner code stored it in the directory structure

    data
    L test_EnvName_DateTime
      L  0
        L train.json
        L train_params.json
      L  1
        L train.json
        L train_params.json
       .
       .
       .
      L  9
        L train.json
        L train_params.json

To plot learning curves from the experiment, averaged over all random
seeds, call

    python plot.py data/test_EnvName_DateTime --value AverageReturn

and voila. To see a different statistics, change what you put in for
the keyword --value. You can also enter /multiple/ values, and it will 
make all of them in order.


Suppose you ran two experiments: 'test1' and 'test2'. In 'test2' you tried
a different set of hyperparameters from 'test1', and now you would like 
to compare them -- see their learning curves side-by-side. Just call

    python plot.py data/test1 data/test2

and it will plot them both! They will be given titles in the legend according
to their exp_name parameters. If you want to use custom legend titles, use
the --legend flag and then provide a title for each logdir.
-------------------------------------------------------------------------------

author: Jiahao Yao

"""

def plot_data(data, value="mean_reward"):
    if isinstance(data, list):
        data = pd.concat(data, ignore_index=True)
        
    logfid = [-np.log10(1 - f) for f in data[value]]# Plot log fidelity
    data['log_Fid'] = logfid
    
    plt.rcParams.update({'font.size': 14})

    # sns.set(style="darkgrid", font_scale=1.5)
    # print(data)
    fig, ax = plt.subplots()
    sns.lineplot(data=data, x="iter", y='log_Fid', hue="Condition") #sns plots 95% confidence interval

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels)
    # plt.legend( loc='best').set_draggable(True)
    # eps = 1e-2
    # plt.ylim(0.0-eps,1.0+eps)
    plt.ylabel(r'$\displaystyle -\log_{10}(1-F(\theta))$')
    # else:
    #     plt.ylabel(r'$\displaystyle F(\theta)$')
    plt.tight_layout()
    plt.show()


# def plot_data_log(data, value="mean_reward"):
#     if isinstance(data, list):
#         data = pd.concat(data, ignore_index=True)

#     # loginfi = pd.Series([np.log10(1 - f) for f in data[value]])# Plot infidelity
#     logfid = [-np.log10(1 - f) for f in data[value]]# Plot log fidelity
#     data['log_Fid'] = logfid

#     sns.set(style="darkgrid", font_scale=1.5)
#     sns.lineplot(data=data, x="iter", value='log_Fid', unit="Unit", condition="Condition")

#     plt.legend( loc='best').set_draggable(True)
#     eps = 1e-2
#     plt.ylabel(r'$\displaystyle -\log_{10}(1-J(\theta_p))$')
#     plt.tight_layout()
#     plt.show()


def get_table(filepath):

    iter_list = []
    loss_list = []
    avgrew_list = []
    maxrew_list = []
    tesrew_list = []
    hisrew_list = []
    ent_list = []

    with open(filepath, 'r') as f:
        for x in f:
            a = re.findall("iter:.*", x)

            if a == []:
                pass
            else:
                iter_list.append( int(a[0].split(',')[0].split(':')[-1].strip()) )
                loss_list.append( float(a[0].split(',')[1].split(':')[-1].strip()) )
                avgrew_list.append( float(a[0].split(',')[2].split(':')[-1].strip()) )
                maxrew_list.append( float(a[0].split(',')[3].split(':')[-1].strip()) )
                tesrew_list.append( float(a[0].split(',')[4].split(':')[-1].strip()) )
                hisrew_list.append( float(a[0].split(',')[5].split(':')[-1].strip()) )
                #ent_list.append( float(a[0].split(',')[6].split(':')[-1].strip()) )


    train_dict = {
        'iter': iter_list,
        'loss': loss_list,
        'mean_reward': avgrew_list,
        'max_reward': maxrew_list,
        'test_reward': tesrew_list, 
        'his_reward': hisrew_list,
        #'entropy': ent_list
    }


    train_df = pd.DataFrame(train_dict)

    return train_df


def get_datasets(fpath, condition=None):
    unit = 0
    datasets = []
    for root, dir, files in os.walk(fpath):
        if 'train.log' in files:
            param_path = open(os.path.join(root,'train_params.json'))
            params = json.load(param_path)

            exp_name = 'batch_size = {}'.format( params['batch_size'])

            log_path = os.path.join(root,'train.log')
            experiment_data = get_table(log_path)

            experiment_data.insert(
                len(experiment_data.columns),
                'Unit',
                unit
                )        
            experiment_data.insert(
                len(experiment_data.columns),
                'Condition',
                condition or exp_name
                )

            datasets.append(experiment_data)
            unit += 1

    return datasets


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('logdir', nargs='*')
    parser.add_argument('--legend', nargs='*')
    parser.add_argument('--value', default='mean_reward', nargs='*')
    args = parser.parse_args()

    use_legend = False
    if args.legend is not None:
        assert len(args.legend) == len(args.logdir), \
            "Must give a legend title for each set of experiments."
        use_legend = True

    data = []
    if use_legend:
        for logdir, legend_title in zip(args.logdir, args.legend):
            data += get_datasets(logdir, legend_title)
    else:
        for logdir in args.logdir:
            data += get_datasets(logdir)

    if isinstance(args.value, list):
        values = args.value
    else:
        values = [args.value]
    for value in values:
        plot_data(data, value=value)

if __name__ == "__main__":
    main()

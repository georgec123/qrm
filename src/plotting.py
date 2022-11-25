import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def print_viols_and_plot(df: pd.DataFrame, title: str, q: str):
    """
    Print summary of data including: 99 VaR violations, 95 VaR violations, and VaR/ES plot
    """
    data = df.copy()
    data.dropna(inplace=True)
    ylabel = "Daily portfolio loss (%) (positive part)"
    
    ################ plot main chart################
    ax = data[['max_loss']].plot(c='orange', linewidth=0.5, figsize=(10, 6))
    data[['var_95','es_95','var_99','es_99']].plot(ax=ax, style=['r--','r-','b--','b-'], linewidth=0.5)


    ax.set_ylabel(ylabel)
    ax.set_xlabel("Time")
    
    months = mdates.MonthLocator((1,4,7,10))
    ax.xaxis.set_minor_locator(months)
    ax.xaxis.tick_bottom()
    ax.yaxis.tick_left()
    
    ax.set_title(title)
    ax.legend(loc='upper left')
    plt.savefig(f'./plots/q2/Q2{q}i.png')

    plt.show()

    ################ plot var viols ################
    for idx, var in enumerate(['95', '99']):
        ax = data[['max_loss']].plot(c='orange', linewidth=0.5, figsize=(10, 6));
        ax = data[[f'var_{var}']].plot(ax=ax, style=['b--'], linewidth=0.5)

        viols = data[data['max_loss']>data[f'var_{var}']]
        ax.scatter(viols.index,viols['max_loss'], marker='o', c='r', s=10, zorder=10)
        ax.set_title(f"{title}. VaR {var}% violations")
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Time")
        plt.savefig(f'./plots/q2/Q2{q}{"i"*(idx+2)}.png')

        plt.show()



    num_days = (~data['var_95'].isna()).sum()
    viols_95 = (data['loss']>data['var_95']).sum()
    viols_99 = (data['loss']>data['var_99']).sum()

    print(f"Violations 95%: {viols_95}, {100*viols_95/num_days:.2f}%")
    print(f"Violations 99%: {viols_99}, {100*viols_99/num_days:.2f}%")
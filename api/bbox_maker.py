
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

from io import BytesIO
from dataclasses import dataclass, field

@dataclass
class BoxPlotMaker():

    describe_dataset_path : str = "data/output/describe.csv"
    describe_df : pd.DataFrame = None

    def __post_init__(self):
        self.describe_df = pd.read_csv(self.describe_dataset_path, index_col=0)


    def build_boxplot(self, loan_series):
        def get_bp_data(col_id, d_df):
            return [ d_df[col_id].loc["min"], d_df[col_id].loc["25%"], d_df[col_id].loc["50%"], d_df[col_id].loc["75%"], d_df[col_id].loc["max"] ]

        matplotlib.use("agg")
        f, axes = plt.subplots(4, 1, figsize=(8,5))

        sns.boxplot(data=get_bp_data("PAYMENT_RATE", self.describe_df), orient="h", palette="mako", width=.17,  ax=axes[0] )
        sns.boxplot(data=get_bp_data("AMT_CREDIT", self.describe_df), orient="h", palette="crest", width=.17, ax=axes[1] )
        sns.boxplot(data=get_bp_data("EXT_SOURCE_1", self.describe_df), orient="h", palette="Set2", width=.17 , ax=axes[2] )
        sns.boxplot(data=get_bp_data("EXT_SOURCE_2", self.describe_df), orient="h", palette="Set2", width=.17, ax=axes[3] )

        sns.swarmplot(data=[loan_series["PAYMENT_RATE"]], orient="h", ax=axes[0], marker='d', size=25, color="red")
        sns.swarmplot(data=[loan_series["AMT_CREDIT"]], orient="h", ax=axes[1], marker='d', size=25, color="red")
        sns.swarmplot(data=[loan_series["EXT_SOURCE_1"]], orient="h", ax=axes[2], marker='d', size=25, color="red")
        sns.swarmplot(data=[loan_series["EXT_SOURCE_2"]], orient="h", ax=axes[3], marker='d', size=25, color="red")

        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_frame_on(False)
        axes[0].set_ylabel("PAYMENT_RATE", rotation="30", fontsize="large", fontweight="bold", fontfamily="monospace")
        axes[1].set_ylabel("AMT_CREDIT", rotation="30", fontsize="large", fontweight="bold", fontfamily="monospace")
        axes[2].set_ylabel("EXT_SOURCE_1", rotation="30", fontsize="large", fontweight="bold", fontfamily="monospace")
        axes[3].set_ylabel("EXT_SOURCE_2", rotation="30", fontsize="large", fontweight="bold", fontfamily="monospace")

        # Create a buffer to store image data
        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)

        

        return buf
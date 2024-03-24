import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from catboost import CatBoostClassifier, cv, Pool


class DataProcessor:
    def __init__(self, file_path):
        self.full_data = pd.read_csv(file_path)
        self.data = self.full_data.drop(columns='INSTN_NAME')
        self.imputed = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.best_cat = None


    def create_target_column(self, percentile):
        threshold = (
            self.data['DROPOUT_7-12 Drop Outs -ALL Students'].quantile(percentile)
        )
        self.data['high_dropout'] = (
            (self.data['DROPOUT_7-12 Drop Outs -ALL Students'] >=
            threshold).astype(int)
        )
        self.data = (
            self.data[[col for col in self.data.columns if 'DROPOUT' not in col]]
        )


    def prepare_data(self):
        imputer = KNNImputer(n_neighbors=5)
        imputed = imputer.fit_transform(self.data)
        self.imputed = pd.DataFrame(imputed, columns=self.data.columns)
        self.X = self.imputed.drop(columns=['high_dropout'])
        self.y = self.imputed['high_dropout']

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )


    def baseline_logreg(self):
        if self.imputed is None:
            self.prepare_data()

        model = LogisticRegression(max_iter=10000).fit(self.X_train, self.y_train)

        y_pred = model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        report = classification_report(self.y_test, y_pred)

        return accuracy, report


    def train_cat(self):
        cv_dataset = Pool(data=self.X, label=self.y)

        params = {
            'iterations': 100,
            'rsm': 0.8,
            'learning_rate': 0.1,
            'depth': 6,
            'loss_function': 'Logloss'
        }

        scores = cv(cv_dataset, params, fold_count=5)
        return scores

    def cat_f1(self, scores):
        best_iteration = max(scores['iterations'])

        best_model = CatBoostClassifier(
            iterations=best_iteration,
            learning_rate=0.1,
            depth=6,
            loss_function="Logloss",
            verbose=0
        )

        best_model.fit(self.X, self.y, verbose=0)

        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

        y_pred = best_model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        report = classification_report(self.y_test, y_pred)

        self.best_cat = best_model

        return accuracy, report 


    def feature_importance_list(self):
        feature_importance = self.best_cat.get_feature_importance()
        feature_names = self.best_cat.feature_names_

        feature_dict = dict(zip(
            feature_names, feature_importance
        ))

        self.sorted_feature_importance = sorted(
            feature_dict.items(), key=lambda x: x[1],
            reverse=True
        )

        return self.sorted_feature_importance


    def cluster_at_risk_schools(self):
        top_10 = [tup[0] for tup in self.sorted_feature_importance[:10]]

        high_drop_schools = self.imputed[self.imputed['high_dropout']==1]

        scaler = MinMaxScaler()
        normalized_data = scaler.fit_transform(high_drop_schools)

        param_grid = {
            'n_clusters': [2, 3, 4, 5],
            'init': ['k-means++', 'random'],
            'max_iter': [100, 200, 300]
        }

        kmeans = KMeans()

        grid_search = GridSearchCV(estimator=kmeans, param_grid=param_grid, cv=5)
        grid_search.fit(normalized_data)

        best_kmeans = grid_search.best_estimator_

        clusters = best_kmeans.fit_predict(normalized_data)

        _tsne = TSNE(n_components=2, random_state=42)
        tsne_data = _tsne.fit_transform(normalized_data)

        df_tsne = pd.DataFrame(tsne_data, columns=['t-SNE Component 1', 't-SNE Component 2'])
        df_tsne['Cluster'] = clusters

        df_tsne = pd.concat([df_tsne, self.full_data[['INSTN_NAME'] + top_10]], axis=1)
        st.write(df_tsne)

        fig = px.scatter(df_tsne, x='t-SNE Component 1', y='t-SNE Component 2', 
                         color='Cluster', title='t-SNE Plot of K-means Clustering', 
                         opacity=0.75, color_continuous_scale='viridis',
                         hover_data=['INSTN_NAME'] + top_10)
        fig.update_layout(xaxis_title='t-SNE Component 1', yaxis_title='t-SNE Component 2', coloraxis_colorbar=dict(title='Cluster'))

        cluster_analysis = high_drop_schools[["SCHOOL_ID"] + top_10].copy()
        cluster_analysis["cluster"] = clusters

        cluster_analysis = cluster_analysis.groupby("cluster").mean()

        stand_scaler = StandardScaler()
        standardized_cluster_analysis = (
            pd.DataFrame(stand_scaler.fit_transform(cluster_analysis),
                        columns=cluster_analysis.columns,
                        index=cluster_analysis.index
                        )
        )

        return fig, standardized_cluster_analysis


def multi_sel_scatter(df):
    selected_columns = st.multiselect(
        'Select Two Columns for Correlation',
        df.columns
    )
    if len(selected_columns) == 2:
        fig = px.scatter(df,
                         x=selected_columns[0],
                         y=selected_columns[1],
                         title=f'Scatterplot of {selected_columns[0]} vs {selected_columns[1]}'
                         )
        st.plotly_chart(fig)


def build_logreg(data_proc):
    st.markdown(
        '''
        ## Logistic Regression
        We use logistic regression as a baseline technique to showcase
        the effectiveness of our model. 
        '''
    )
    with st.expander("Baseline Logistic Regression Performance"):
        accuracy, report = data_proc.baseline_logreg()

        st.write("Below is the score of our logistic regression. \
            We seek to beat this for our model.")
        st.markdown("##### Accuracy: ")
        st.write(accuracy)
        st.markdown("##### Classification Report")
        st.text(report)


def build_cat(data_proc):
    st.markdown(
        '''
        ## CatBoost
        This is our chosen model, CatBoost. 
        CatBoost utilizes gradient boosted decision trees to model a classify the target variable.
        '''
    )
    with st.expander("Catboost Model Performance"):
        scores = data_proc.train_cat()
        st.write("These are our Model\'s scores over a many iterations while being cross-validated.")
        st.write(scores)

        accuracy, report = data_proc.cat_f1(scores)
        st.write("These are our scores to compare to logistic regression.")
        st.markdown("##### Accuracy: ")
        st.write(accuracy)
        st.markdown("##### Classification Report")
        st.text(report)

        st.write("This is our list of features and how important they are to classifying dropout rate.")
        sorted_feat_imp = data_proc.feature_importance_list()
        st.dataframe(pd.DataFrame(sorted_feat_imp,
                                  columns=['Features', 'Importance']))

def explore_clusters(data_proc):
    with st.expander("Segmentation Analysis"):
        st.write("These are our clusters with our top 10 features. This groups our at risk schools into several categories.")
        fig, cluster_analysis = data_proc.cluster_at_risk_schools()
        st.write("This is the visual representation of our clusters. This 10 dimensional representation has been reduced to 2 for interpretability.")
        st.write("You can hover over individual dots to see more information about them. (It is a little buggy, at the moment.)")
        st.plotly_chart(fig)
        st.write("This is the statistical representation of the clusters. We seek to qualitatively describe these clusters using background infromation.")
        st.write(cluster_analysis.drop(columns='SCHOOL_ID'))


def main():
    st.title('At Risk Schools')
    st.markdown(
        '''
        This is an attempt to identify at risk schools using dropout rates
        as a target variable. Schools with a dropout rate of above the 75th
        percentil are identified as at risk.
        '''
    )
    data_processor = (
        DataProcessor('Focused_School_Data_Cleaned.csv')
    )
    data_processor.create_target_column(0.75)
    with st.expander("Exploratory Data Analysis"):
        st.write('This is our dataset. It is a list of schools with many features that we seek to learn from.')
        st.dataframe(data_processor.full_data)
        st.write("This is the statistical distribution of each of our columns.")
        st.dataframe(data_processor.data.drop(columns='SCHOOL_ID').describe())
        st.write("This is the correlation between two columns.")
        st.dataframe(data_processor.data.drop(columns='SCHOOL_ID').corr())
        st.write("Here you can select two columns and a graph will show a scatterplot of the two features.")
        multi_sel_scatter(data_processor.data)
    build_logreg(data_processor)
    build_cat(data_processor)
    explore_clusters(data_processor)


if __name__ == '__main__':
    main()

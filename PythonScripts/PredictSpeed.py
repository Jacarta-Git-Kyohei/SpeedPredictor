import os
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_SAVED_FOLDER = os.path.join(SCRIPT_DIR, "..", "CreatedDataset")
SAVED_ERROR_DISTRIBUTION_CSV_FOLDER = os.path.join(SCRIPT_DIR, "..", "ErrorDistributionCSV")

# ------ Customize Parameters ------
DATASET_FILE_NAME = "SpeedPredictorDataset_Chunk3500.csv"
bDisplayErrorDistribution_LR = False
bDisplayErrorDistribution_SVR = False
bDisplayErrorDistribution_XGB = False
bDisplayErrorDistribution_Ensemble = False
bSaveCSVErrorDistribution_LR = False
bSaveCSVErrorDistribution_SVR = False
bSaveCSVErrorDistribution_XGB = True
bSaveCSVErrorDistribution_Ensemble = True
bDisplayImportanceOfX_LR = False
bDisplayImportanceOfX_SVR = False
bDisplayImportanceOfX_XGB = False
#-----------------------------------

# ------ Machine Learning ------
def GetPred_LinearRegression(X_Train, Y_Train, X_Test):
    Model_LR = LinearRegression()
    Model_LR.fit(X_Train, Y_Train)
    PredictedVals_LR = Model_LR.predict(X_Test)
    return PredictedVals_LR

def GetPred_SVR(X_Train, Y_Train, X_Test):
    Model_SVR = SVR(kernel="rbf")
    Model_SVR.fit(X_Train, Y_Train)
    PredictedVals_SVR = Model_SVR.predict(X_Test)
    return PredictedVals_SVR

def GetPred_XGBoost(X_Train, Y_Train, X_Test):
    Model_XGB = XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    objective="reg:squarederror",
    random_state=42
    )
    Model_XGB.fit(X_Train, Y_Train)
    PredictedVals_XGB = Model_XGB.predict(X_Test)
    return PredictedVals_XGB
# ------------------------------

# ------ Evaluation ------
def GetEvaluation(y_true, y_pred):
    me   = np.mean(y_pred - y_true)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    return round(me, 2), round(mae, 2), round(rmse, 2), round(r2, 2)

def PlotErrorDistribution(y_true, y_pred, model_name):
    residuals = y_pred - y_true

    # Histogram
    plt.figure(figsize=(8,6))
    plt.hist(residuals, bins=30, color='skyblue', edgecolor='black')
    plt.xlabel("Prediction Error (Predicted - True)")
    plt.ylabel("Count")
    plt.title(f"{model_name} Residuals Histogram")
    plt.show()

    # Pred vs True
    plt.figure(figsize=(8,6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()],[y_true.min(), y_true.max()], 'r--')
    plt.xlabel("True Speed (km/h)")
    plt.ylabel("Predicted Speed (km/h)")
    plt.title(f"{model_name} Predicted vs True")
    plt.show()

    # Boxplot
    plt.figure(figsize=(6,6))
    plt.boxplot(residuals, vert=True)
    plt.ylabel("Prediction Error")
    plt.title(f"{model_name} Residuals Boxplot")
    plt.show()

def ComputeAndPlotSHAP(model, X_train, X_test, model_name):
    feature_names = X_train.columns
    if isinstance(model, LinearRegression):
        explainer = shap.LinearExplainer(model, X_train)
        shap_values = explainer.shap_values(X_test)
    elif isinstance(model, SVR):
        # SVR calculations are computationally intensive, so only 200 cases will be processed.
        X_sample = shap.sample(X_train, 200, random_state=42)
        explainer = shap.KernelExplainer(model.predict, X_sample)
        shap_values = explainer.shap_values(X_test, nsamples=200)
    else:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

    shap.summary_plot(shap_values, X_test, feature_names=feature_names, plot_type="bar")
    shap.summary_plot(shap_values, X_test, feature_names=feature_names)

def SaveResidualsToCSV(y_true, y_pred, model_name):
    residuals = y_pred - y_true
    df = pd.DataFrame({
        "True": y_true,
        "Predicted": y_pred,
        "Residual": residuals
    })
    csv_path = os.path.join(SAVED_ERROR_DISTRIBUTION_CSV_FOLDER, f"{model_name}_Residuals.csv")
    df.to_csv(csv_path, index=False)
    print(f"{model_name} residuals saved to {csv_path}")
# ------------------------

def main():

    DataFrame = pd.read_csv(os.path.join(OUTPUT_SAVED_FOLDER, DATASET_FILE_NAME))

    # Explanatory variable X : Response variable Y = 8 : 2
    X = DataFrame[['accel_STD', 'accel_RMS', 'accel_DomFreq1', 'accel_DomFreq2',
                'ang_STD', 'ang_RMS', 'ang_DomFreq1', 'ang_DomFreq2']]
    Y = DataFrame['TargetSpeed']
    X_Train, X_Test, Y_Train, Y_Test = train_test_split(
        X, Y, test_size=0.2, shuffle=True, random_state=20
    )

    # Predict by Machine Learning
    PredictedVals_LR = GetPred_LinearRegression(X_Train, Y_Train, X_Test)
    PredictedVals_SVR = GetPred_SVR(X_Train, Y_Train, X_Test)
    PredictedVals_XGB = GetPred_XGBoost(X_Train, Y_Train, X_Test)
    PredictedVals_Ensemble = (PredictedVals_LR + PredictedVals_SVR + PredictedVals_XGB) / 3.0

    print("===== Regression Evaluation =====")
    me, mae, rmse, r2 = GetEvaluation(Y_Test, PredictedVals_LR)
    print(f"LinearRegression | (me, mae, rmse, r2) = ({me}, {mae}, {rmse}, {r2})")
    me, mae, rmse, r2 = GetEvaluation(Y_Test, PredictedVals_SVR)
    print(f"SVR              | (me, mae, rmse, r2) = ({me}, {mae}, {rmse}, {r2})")
    me, mae, rmse, r2 = GetEvaluation(Y_Test, PredictedVals_XGB)
    print(f"XGBoost          | (me, mae, rmse, r2) = ({me}, {mae}, {rmse}, {r2})")
    me, mae, rmse, r2 = GetEvaluation(Y_Test, PredictedVals_Ensemble)
    print(f"Ensemble         | (me, mae, rmse, r2) = ({me}, {mae}, {rmse}, {r2})")

    # Error Distribution
    if bDisplayErrorDistribution_LR:
        PlotErrorDistribution(Y_Test, PredictedVals_LR,  "Linear Regression")
    if bDisplayErrorDistribution_SVR:
        PlotErrorDistribution(Y_Test, PredictedVals_SVR, "SVR")
    if bDisplayErrorDistribution_XGB:
        PlotErrorDistribution(Y_Test, PredictedVals_XGB, "XGBoost")
    if bDisplayErrorDistribution_Ensemble:
        PlotErrorDistribution(Y_Test, PredictedVals_Ensemble, "Ensemble")

    if bSaveCSVErrorDistribution_LR:
        SaveResidualsToCSV(Y_Test, PredictedVals_LR, "LinearRegression")
    if bSaveCSVErrorDistribution_SVR:
        SaveResidualsToCSV(Y_Test, PredictedVals_SVR, "SVR")
    if bSaveCSVErrorDistribution_XGB:
        SaveResidualsToCSV(Y_Test, PredictedVals_XGB, "XGBoost")
    if bSaveCSVErrorDistribution_Ensemble:
        SaveResidualsToCSV(Y_Test, PredictedVals_Ensemble, "Ensemble")

    # SHAP Importance
    if bDisplayImportanceOfX_LR:
        Model_LR  = LinearRegression().fit(X_Train, Y_Train)
        ComputeAndPlotSHAP(Model_LR,  X_Train, X_Test, "Linear Regression")
    if bDisplayImportanceOfX_SVR:
        Model_SVR = SVR(kernel="rbf").fit(X_Train, Y_Train)
        ComputeAndPlotSHAP(Model_SVR, X_Train, X_Test, "SVR")
    if bDisplayImportanceOfX_XGB:
        Model_XGB_imp = XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            objective="reg:squarederror",
            random_state=42
        ).fit(X_Train, Y_Train)
        ComputeAndPlotSHAP(Model_XGB_imp, X_Train, X_Test, "XGBoost")


if __name__ == "__main__":
    main()

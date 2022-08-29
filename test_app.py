import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor

st.write("""
# Sample APP In Progress

""")

st.write('---')

# Loads the Boston House Price Dataset
tumor = datasets.load_breast_cancer()
X = pd.DataFrame(tumor.data, columns=tumor.feature_names)
Y = pd.DataFrame(tumor.target, columns=["target"])
X.columns = ["Var%d" % i for i,_ in enumerate(X.columns)]
X.rename(columns = {'Var30':'target'}, inplace = True)
X = X.loc[0:15]
Y = Y.loc[0:15]


# Sidebar
# Header of Specify Input Parameters
st.sidebar.header('Values are set at `mean` \nSpecify Input Parameters')

def user_input_features():
    Var0 = st.sidebar.slider('Var0', X.Var0.min(), X.Var0.max(), X.Var0.mean())
    Var1 = st.sidebar.slider('Var1', X.Var1.min(), X.Var1.max(), X.Var1.mean())
    Var2 = st.sidebar.slider('Var2', X.Var2.min(), X.Var2.max(), X.Var2.mean())
    Var3 = st.sidebar.slider('Var3', X.Var3.min(), X.Var3.max(), X.Var3.mean())
    Var4 = st.sidebar.slider('Var4', X.Var4.min(), X.Var4.max(), X.Var4.mean())
    Var5 = st.sidebar.slider('Var5', X.Var5.min(), X.Var5.max(), X.Var5.mean())
    Var6 = st.sidebar.slider('Var6', X.Var6.min(), X.Var6.max(), X.Var6.mean())
    Var7 = st.sidebar.slider('Var7', X.Var7.min(), X.Var7.max(), X.Var7.mean())
    Var8 = st.sidebar.slider('Var8', X.Var8.min(), X.Var8.max(), X.Var8.mean())
    Var9 = st.sidebar.slider('Var9', X.Var9.min(), X.Var9.max(), X.Var9.mean())
    Var10 = st.sidebar.slider('Var10', X.Var10.min(), X.Var10.max(), X.Var10.mean())
    Var11 = st.sidebar.slider('Var11', X.Var11.min(), X.Var11.max(), X.Var11.mean())
    Var12 = st.sidebar.slider('Var12', X.Var12.min(), X.Var12.max(), X.Var12.mean())
    Var13 = st.sidebar.slider('Var13', X.Var13.min(), X.Var13.max(), X.Var13.mean())
    Var14 = st.sidebar.slider('Var14', X.Var14.min(), X.Var14.max(), X.Var14.mean())
    Var15 = st.sidebar.slider('Var15', X.Var15.min(), X.Var15.max(), X.Var15.mean())
    Var16 = st.sidebar.slider('Var16', X.Var16.min(), X.Var16.max(), X.Var16.mean())
    Var17 = st.sidebar.slider('Var17', X.Var17.min(), X.Var17.max(), X.Var17.mean())
    Var18 = st.sidebar.slider('Var18', X.Var18.min(), X.Var18.max(), X.Var18.mean())
    Var19 = st.sidebar.slider('Var19', X.Var19.min(), X.Var19.max(), X.Var19.mean())
    Var20 = st.sidebar.slider('Var20', X.Var20.min(), X.Var20.max(), X.Var20.mean())
    Var21 = st.sidebar.slider('Var21', X.Var21.min(), X.Var21.max(), X.Var21.mean())
    Var22 = st.sidebar.slider('Var22', X.Var22.min(), X.Var22.max(), X.Var22.mean())
    Var23 = st.sidebar.slider('Var23', X.Var23.min(), X.Var23.max(), X.Var23.mean())
    Var24 = st.sidebar.slider('Var24', X.Var24.min(), X.Var24.max(), X.Var24.mean())
    Var25 = st.sidebar.slider('Var25', X.Var25.min(), X.Var25.max(), X.Var25.mean())
    Var26 = st.sidebar.slider('Var26', X.Var26.min(), X.Var26.max(), X.Var26.mean())
    Var27 = st.sidebar.slider('Var27', X.Var27.min(), X.Var27.max(), X.Var27.mean())
    Var28 = st.sidebar.slider('Var28', X.Var28.min(), X.Var28.max(), X.Var28.mean())
    Var29 = st.sidebar.slider('Var29', X.Var29.min(), X.Var29.max(), X.Var29.mean())
    data = {'Var0': Var0,
            'Var1': Var1,
            'Var2': Var2,
            'Var3': Var3,
            'Var4': Var4,
            'Var5': Var5,
            'Var6': Var6,
            'Var7': Var7,
            'Var8': Var8,
            'Var9': Var9,
            'Var10': Var10,
            'Var11': Var11,
            'Var12': Var12,
            'Var13': Var13,
            'Var14': Var14,
            'Var15': Var15,
            'Var16': Var16,
            'Var17': Var17,
            'Var18': Var18,
            'Var19': Var19,
            'Var20': Var20,
            'Var21': Var21,
            'Var22': Var22,
            'Var23': Var23,
            'Var24': Var24,
            'Var25': Var25,
            'Var26': Var26,
            'Var27': Var27,
            'Var28': Var28,
            'Var29': Var29
            }
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

# Main Panel

# Print specified input parameters
st.header('Specified Input parameters')
st.write(df)
st.write('---')

# Build Regression Model
model = RandomForestRegressor()
model.fit(X, Y)
# Apply Model to Make Prediction
prediction = model.predict(df)

st.header('Prediction of `target`')
st.write(prediction)
st.write('---')

# # Explaining the model's predictions using SHAP values
# # https://github.com/slundberg/shap
# explainer = shap.TreeExplainer(model)
# shap_values = explainer.shap_values(X)
# 
# st.header('Feature Importance')
# plt.title('Feature importance based on SHAP values')
# shap.summary_plot(shap_values, X)
# st.pyplot(bbox_inches='tight')
# st.write('---')
# 
# plt.title('Feature importance based on SHAP values (Bar)')
# shap.summary_plot(shap_values, X, plot_type="bar")
# st.pyplot(bbox_inches='tight')

import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import f, chi2
from numpy import pi, sin, cos
import plotly.graph_objects as go
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler

########## To run the app type: streamlit run PLS_SL3.py


# To save data and not execute the same operation


#---------------------------------#
# Page layout
## Page expands to full width
st.set_page_config(page_title='The PLS App', layout='wide')

#---------------------------------#
# Model building


def ellipse(x_center=0, y_center=0, ax1 = [1, 0],  ax2 = [0,1], a=1, b =1,  N=360):
    # x_center, y_center the coordinates of ellipse center
    # ax1 ax2 two orthonormal vectors representing the ellipse axis directions
    # a, b the ellipse parameters
    if np.linalg.norm(ax1) != 1 or np.linalg.norm(ax2) != 1:
        raise ValueError('ax1, ax2 must be unit vectors')
    if  abs(np.dot(ax1, ax2)) > 1e-06:
        raise ValueError('ax1, ax2 must be orthogonal vectors')
    t = np.linspace(0, 2*pi, N)
    #ellipse parameterization with respect to a system of axes of directions a1, a2
    xs = a * cos(t)
    ys = b * sin(t)
    #rotation matrix
    R = np.array([ax1, ax2]).T
    # coordinate of the  ellipse points with respect to the system of axes [1, 0], [0,1] with origin (0,0)
    xp, yp = np.dot(R, [xs, ys])
    x = xp + x_center 
    y = yp + y_center
    return x, y


def obs_comp_plot(axis_x, axis_y, color_select):  
    # Plotting the first two components
    a = np.sqrt(var[axis_x-1] * crit_99)
    b = np.sqrt(var[axis_y-1] * crit_99)
    fig = go.Figure()
    fig.add_hline(y=0)
    fig.add_vline(x=0)
    x_ellipse, y_ellipse = ellipse(a=a, b=b)

    fig.add_trace(go.Scatter(x=x_ellipse, y=y_ellipse, mode = 'lines'))
    fig.add_trace(go.Scatter(x=pls.x_scores_[:,axis_x-1], y=pls.x_scores_[:,axis_y-1], marker_color=color_select, 
                             marker_showscale=True, mode='markers', hovertemplate=df.index, name=''))
    fig.update_layout(width =700, height=700)
    st.plotly_chart(fig)


def var_comp_plot(axis_x, axis_y):
    fig = go.Figure()
    fig.add_hline(y=0)
    fig.add_vline(x=0)
    fig.add_trace(go.Scatter(x=pls.x_rotations_[:,axis_x-1], y=pls.x_rotations_[:,axis_y-1], mode='markers', text=df.columns, name='Data'))
    fig.update_layout(width =800, height=800)
    st.plotly_chart(fig)


def var_contrib():
    fig = go.Figure()
    fig.add_trace( go.Bar( x=df.columns[0:-1], y=pls.coef_.T[0]) )
    fig.update_layout(width =800, height=600)
    st.plotly_chart(fig)


def T2_plot():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(N), y=T_hot, mode='markers+lines', text=df.index, name='Data'))
    fig.add_hline(y=crit_99, line_dash="dash", line_color="red", annotation_text="Crit 99%: "+str(round(crit_99, 3)), 
                  annotation_position="bottom right")
    fig.add_hline(y=crit_99*2, line_dash="dash", line_color="red", annotation_text="2 * Crit 99%" , 
                  annotation_position="bottom right")
    fig.update_layout(width =800, height=600)
    st.plotly_chart(fig)


def SPE_plot():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(N), y=SCR, mode='markers+lines', text=df.index, name='Data'))
    fig.add_hline(y=spe_crit, line_dash="dash", line_color="red", annotation_text="Crit 99%: "+str(round(spe_crit, 3)), 
                  annotation_position="bottom right")
    fig.update_layout(width =800, height=600)
    st.plotly_chart(fig)


def obs_contrib(contrib):
    fig = go.Figure()
    fig.add_trace( go.Bar( x=df.columns[0:-1], y=contrib) )
    st.plotly_chart(fig)


def SPE_contrib_plot():
    SPE_index = index_df.loc[obs_for_contrib, 'Loc'].values
    SPE_contrib = np.sum( np.square(X[SPE_index] - X_est[SPE_index]) ,axis=0) / len(obs_for_contrib)
    fig = go.Figure()
    fig.add_trace( go.Bar( x=df.columns[0:-1], y=SPE_contrib) )
    st.markdown('**4.1 SPE Contributions per variable**')
    st.plotly_chart(fig)


def T2_contrib_plot():
    # Components
    T2_index = index_df.loc[obs_for_contrib, 'Loc'].values

    T2_contrib_per_comp = np.sum( np.square(t[T2_index]) / var, axis=0) / len(obs_for_contrib)
    fig = go.Figure()
    fig.add_trace( go.Bar( x=['C' + str(i+1) for i in range(A)], y=T2_contrib_per_comp) )
    st.markdown('**4.1 T^2 Contributions per component**')
    st.plotly_chart(fig)

    # Variables
    if len(select_comp_T2_contrib) == 0:
        st.info("Select components for detail per variable")
        return

    comp_index = [i-1 for i in select_comp_T2_contrib]

    M = np.zeros((len(T2_index), P, len(select_comp_T2_contrib)))
    x_ij = X[T2_index]
    scores_ik = scores[T2_index][:,comp_index]
    loadings_jk = loadings[:,comp_index]
    var_k = var[comp_index]
    
    # Obs i, variable j, component k
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            for k in range(M.shape[2]):
                M[i,j,k] = scores_ik[i,k] / var_k[k] * loadings_jk[j,k] * x_ij[i,j] 

    T2_contrib_variab = np.sum(np.sum(M, axis=2), axis=0)
    fig2 = go.Figure()
    fig2.add_trace( go.Bar( x=df.columns[0:-1], y=T2_contrib_variab) )
    st.markdown('**4.2 T^2 Contributions per variable for selected components**')
    st.plotly_chart(fig2)



@st.cache
def get_data(df):
    data = df.to_numpy()
    X = data[:, 0:-1]
    y = data[:, -1]
    N = X.shape[0]
    P = X.shape[1]
    return data, X, y, N, P

@st.cache
def build_model(X, y, N, A):
    sc = StandardScaler()
    X = sc.fit_transform(X)
    y = y.reshape(-1,1)
    y = sc.fit_transform(y)

    pls = PLSRegression(n_components= A )
    pls.fit(X, y)

    # variance of each latent value
    var = np.var(pls.x_scores_, axis=0, ddof=1)

    # scores in X
    t = pls.x_scores_

    T_hot = np.sum(np.square(t) / var, axis=1)
    crit_99 = f.ppf(0.99, A, N-A) * A * (N**2 - 1) / (N * (N - A))

    X_est = np.matmul(pls.x_scores_, pls.x_loadings_.T)
    SCR = np.sum(np.square(X - X_est),axis=1)

    g = np.var(SCR, ddof=1) / (2*np.mean(SCR))
    h = 2 * np.mean(SCR)**2 / np.var(SCR, ddof=1)
    spe_crit = g * chi2.ppf(0.99, h) 
    return X, y, pls, var, t, T_hot, crit_99, X_est, SCR, spe_crit

#---------------------------------#
# Display

st.title('PLS App V3')

st.markdown("""
This app performs a PLS for the data selected. It can plot the components, T2 HOtelling and SPE to analyse the observations.
""")

st.sidebar.header('User Input Features')

# Sidebar - Collects user input features into dataframe
with st.sidebar.header('CSV data'):
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])


#---------------------------------#
# Analysis

if uploaded_file is None:
    st.info('Awaiting for CSV file to be uploaded.')
    'Recommended layout for database:'
    '- First column for index'
    '- Last column for Y values'
    '- Middle columns for X varaibles'
    st.stop()


############ All the variables used for plots
df = pd.read_csv(uploaded_file)
with st.sidebar.header('Variables'):
    choose_index = st.sidebar.checkbox('Change index')
    index_obs = st.sidebar.selectbox('Choose new index', df.columns)
    x_vars = st.sidebar.multiselect('X variables', df.columns, default=list(df.columns[1:-1]))
    y_vars = st.sidebar.multiselect('Y variables', df.columns, default=df.columns[-1])

    if choose_index and index_obs != None:
        df.set_index(index_obs, inplace=True)  

data, X, y, N, P = get_data(df)
index_df = pd.DataFrame(np.arange(N),index=df.index,columns=['Loc'])

with st.sidebar.header('Components'):
    A = st.sidebar.slider('Number', 1, P, value=2)

# For now build only with numerical values
try:
    X, y, pls, var, t, T_hot, crit_99, X_est, SCR, spe_crit = build_model(X, y, N, A)
except ValueError:
    st.error('Use numerical values only')
    df2 = pd.DataFrame(df.dtypes.values, index=df.columns.values)
    df2[0] = df2[0].astype('string')
    st.write('Check data types for all variables')
    st.write(df2)
    st.stop()


############
# Expand sidebar options
# Specify parameter settings

# Main panel
st.subheader('1. Dataset')
st.markdown('**1.1 Table**')
st.write(df)

st.subheader('2. Components')
st.markdown('**2.1 Scores plot**')
col1, col2 = st.columns([2,1])
with col2:
    axis_x = st.selectbox('X axis', np.arange(1, A+1), index=0)
    axis_y = st.selectbox('Y axis', np.arange(1, A+1), index=1)
    if st.checkbox('Add colour'):
        color_select = df[st.selectbox('Variable', df.columns)]
    else:
        color_select = None
with col1:
    obs_comp_plot(axis_x, axis_y, color_select)

st.markdown('**2.2 Variables plot**')
var_comp_plot(axis_x, axis_y)
st.markdown('**2.3 Coefficients plot**')
var_contrib()

st.subheader('3. Validation')
st.markdown('**3.1 SPE**')
SPE_plot()
st.markdown('**3.2 T squared Hotelling**')
T2_plot()


loadings = pls.x_loadings_
scores = pls.x_scores_

with st.sidebar.header('Contributions'):
    # form_contrib = st.form('Contributions') later change to form
    statistic_contrib = st.sidebar.selectbox('Statistic',['None','SPE','T^2'])
    obs_for_contrib = st.sidebar.multiselect('Observations', df.index)
    if statistic_contrib == 'T^2':
        select_comp_T2_contrib = st.sidebar.multiselect('Select components', np.arange(1, A+1))
    

lamb = np.zeros((A,A))
np.fill_diagonal(lamb, var)
lamb_inv = np.linalg.inv(lamb)
lamb_inv2 = np.linalg.inv(np.sqrt(lamb))
P_2 = np.matmul(loadings.T, loadings)

st.subheader('4. Contributions')
if statistic_contrib != 'None' and len(obs_for_contrib) > 0:
    if statistic_contrib == 'SPE':
        SPE_contrib_plot()
    else:
        T2_contrib_plot()
else:
    st.info('Select a statistic and observations to calculate contributions')

# Pending: VIP and variable relevance. Extra graph configurator. Modify existing graph (colors, labels, etc.)

st.caption('Created by Matias Risso')

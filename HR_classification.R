library(tidymodels)
library(visdat)
library(tidyr)
library(car)
library(pROC)
library(ggplot2)
library(vip)
library(rpart.plot)
library(DALEXtra)
### Setting up directory and combining datasets for data prep ---------------
setwd("D:\\IITK Data Analytics\\R\\HumanResources-Classification-R-Project\\")
df_train=read.csv('hr_train.csv')
df_test=read.csv('hr_test.csv')
vis_dat(df_train) #no misssing values!

df_test$left=NA 
df_train$data='train' #creating placeholders
df_test$data='test'   #creating placeholders

df=rbind(df_train,df_test)
vis_dat(df)

#### DATA PREPARATION---------------------------
#changing categorical variables to numerical
df$left=as.factor(df$left) #treating response  column separately

dp_pipe=recipe(left~.,data=df) %>%
  

  update_role(sales,salary ,new_role="to_dummies") %>% 
  
  step_unknown(has_role("to_dummies"),new_level="__missing__") %>% 
  step_other(has_role("to_dummies"),threshold =0.02,other="__other__") %>% 
  step_dummy(has_role("to_dummies")) %>%
  step_impute_median(all_numeric(),-all_outcomes())

dp_pipe=prep(dp_pipe)

prep_df=bake(dp_pipe,new_data=NULL)
#test=bake(dp_pipe,new_data=test)

vis_dat(prep_df) #all looks good!

#dropping NA response rows in training set
prep_df=prep_df[!((is.na(prep_df$left)) & prep_df$data=='train'), ]


#separating train and test dataset---------------------------
train=prep_df %>% filter(data=='train') %>% select(-data)
test=prep_df %>% filter(data=='test') %>% select(-data,-left)
vis_dat(train)
vis_dat(test)


#MODEL BUILDING STARTS ------------------------------------------

### DECISION TREE -------------------------------
tree_model=decision_tree( 
  cost_complexity = tune(), 
  tree_depth = tune(),
  min_n = tune() #min number of obs in a node
) %>%
  set_engine("rpart") %>% #package name rpart , show_engines("decision_tree")
  set_mode("classification") #regression/classification


folds = vfold_cv(train, v = 10)


tree_grid = grid_regular(cost_complexity(), tree_depth(),   # run each ot these indvidually to get idea of range
                         min_n(), levels = 5) #select 3 best values for each of these

# below code runs your code on parallel cores of your cpu
# makes the process faster
# doParallel::registerDoParallel() 


my_res=tune_grid(
  tree_model,
  left~.,
  resamples = folds,
  grid = tree_grid,
  metrics = metric_set(roc_auc), #rmse ,mae for regression
  control = control_grid(verbose = TRUE)
)

autoplot(my_res)+theme_light() #roc higher the better

fold_metrics=collect_metrics(my_res) 

my_res %>% show_best()

final_tree_fit=tree_model %>% 
  finalize_model(select_best(my_res)) %>% 
  fit(left~.,data=train)

# feature importance
final_tree_fit %>%
  vip(geom = "col", aesthetics = list(fill = "midnightblue", alpha = 0.8)) +
  scale_y_continuous(expand = c(0, 0))

# plot the tree
rpart.plot(final_tree_fit$fit)

# predictions
train_pred=predict(final_tree_fit,new_data = train,type="prob") %>% select(.pred_1)
test_pred=predict(final_tree_fit,new_data = test,type="prob") %>% select(.pred_1)
colnames(test_pred)='left'
write.csv(test_pred,'Decision Tree.csv',row.names = F)


### finding cutoff for hard classes
train.score=train_pred$.pred_1
real=train$left

# KS plot
rocit = ROCit::rocit(score = train.score, 
                   class = real) 
kplot=ROCit::ksplot(rocit,legend=F)

# cutoff on the basis of KS
my_cutoff=kplot$`KS Cutoff`

## test hard classes 
test_hard_class=as.numeric(test_pred>my_cutoff)



### Random Forest-------------------------------

rf_model = rand_forest(
  mtry = tune(),
  trees = tune(),
  min_n = tune()
) %>%
  set_mode("classification") %>%
    set_engine("ranger")

folds = vfold_cv(train, v = 10)

rf_grid = grid_regular(mtry(c(5,25)), trees(c(100,500)),
                       min_n(c(2,10)),levels = 5)


my_res=tune_grid(
  rf_model,
  left~.,
  resamples = folds,
  grid = rf_grid,
  metrics = metric_set(roc_auc),
  control = control_grid(verbose = TRUE)
)

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               autoplot(my_res)+theme_light()

fold_metrics=collect_metrics(my_res)

my_res %>% show_best()

final_rf_fit=rf_model %>% 
  set_engine("ranger",importance='permutation') %>% 
  finalize_model(select_best(my_res,"roc_auc")) %>% 
  fit(left~.,data=train)

# variable importance 

final_rf_fit %>%
  vip(geom = "col", aesthetics = list(fill = "midnightblue", alpha = 0.8)) +
  scale_y_continuous(expand = c(0, 0))

# predicitons

train_pred=predict(final_rf_fit,new_data = train,type="prob") %>% select(.pred_1)
test_pred=predict(final_rf_fit,new_data = test,type="prob") %>% select(.pred_1)
colnames(test_pred)='left'
write.csv(test_pred,'Random Forest.csv',row.names = F)
### finding cutoff for hard classes

train.score=train_pred$.pred_1

real=train$left

# KS plot

rocit = ROCit::rocit(score = train.score, 
                     class = real) 

kplot=ROCit::ksplot(rocit)

# cutoff on the basis of KS

my_cutoff=kplot$`KS Cutoff`

## test hard classes 

test_hard_class=as.numeric(test_pred>my_cutoff)

## partial dependence plots
model_explainer =explain_tidymodels(
  final_rf_fit,
  data = dplyr::select(train, -left),
  y = as.integer(train$left),
  verbose = FALSE
)

pdp = model_profile(
  model_explainer,
  variables = "family_income",
  N = 2000,
  groups='children'
)

plot(pdp)




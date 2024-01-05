import os
from reviewprediction import logger                           
import pandas as pd
import numpy as np
from datetime import datetime
from reviewprediction.entity.config_entity import DataValidationConfig

class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config
        orders_dataset= pd.read_csv(os.path.join("artifacts/data_ingestion","olist_orders_dataset.csv"))
        order_items_dataset= pd.read_csv(os.path.join("artifacts/data_ingestion","olist_order_items_dataset.csv"))
        order_reviews_dataset= pd.read_csv(os.path.join("artifacts/data_ingestion","olist_order_reviews_dataset.csv"))
        products_dataset= pd.read_csv(os.path.join("artifacts/data_ingestion","olist_products_dataset.csv"))
        order_payments_dataset= pd.read_csv(os.path.join("artifacts/data_ingestion","olist_order_payments_dataset.csv"))
        customers_dataset= pd.read_csv(os.path.join("artifacts/data_ingestion","olist_customers_dataset.csv"))
        geolocation_dataset= pd.read_csv(os.path.join("artifacts/data_ingestion","olist_geolocation_dataset.csv"))
        sellers_dataset= pd.read_csv(os.path.join("artifacts/data_ingestion","olist_sellers_dataset.csv"))
        product_category_name_translation = pd.read_csv(os.path.join("artifacts/data_ingestion","product_category_name_translation.csv"))

        order_items_products = pd.merge(order_items_dataset,products_dataset,on='product_id')
        order_items_products_sellers = pd.merge(order_items_products,sellers_dataset,on='seller_id')
        two_order_items_products_sellers = pd.merge(order_items_products_sellers,orders_dataset,on='order_id')
        two_order_items_products_sellers_customer = pd.merge(two_order_items_products_sellers,customers_dataset,on='customer_id')
        two_order_items_products_sellers_customer_reviews = pd.merge(two_order_items_products_sellers_customer,order_reviews_dataset,on='order_id')
        final_dataframe = pd.merge(two_order_items_products_sellers_customer_reviews,order_payments_dataset,on='order_id')

        # Create a mapping dictionary
        mapping = dict(zip(product_category_name_translation['product_category_name'],
                   product_category_name_translation['product_category_name_english']))

        # Apply the mapping to the 'product_category_name' column in final_dataframe
        final_dataframe['product_category_name'] = final_dataframe['product_category_name'].map(mapping)

        final_dataframe = final_dataframe.drop_duplicates(subset=['order_id','order_purchase_timestamp','product_id','customer_unique_id','review_comment_message'])
        final_dataframe.drop(['order_id','product_id','seller_id','customer_unique_id'], axis=1, inplace=True)
        final_dataframe.dropna(subset=['shipping_limit_date','order_purchase_timestamp','order_delivered_carrier_date','order_delivered_customer_date','order_estimated_delivery_date'], inplace=True)
        intermediate_time = final_dataframe['order_delivered_customer_date'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S").date()) - final_dataframe['order_purchase_timestamp'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S").date())
        final_dataframe['purchase-delivery difference'] = intermediate_time.apply(lambda x:x.days)
        intermediate_time = final_dataframe['order_estimated_delivery_date'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S").date()) - final_dataframe['order_delivered_customer_date'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S").date())
        final_dataframe['estimated-actual delivery difference'] = intermediate_time.apply(lambda x:x.days)

        final_dataframe['product_category_name'].fillna(value=final_dataframe['product_category_name'].mode()[0], inplace=True)
        final_dataframe['product_name_lenght'].fillna(value=final_dataframe['product_name_lenght'].mode()[0], inplace=True)
        final_dataframe['product_description_lenght'].fillna(value=final_dataframe['product_description_lenght'].median(), inplace=True)
        final_dataframe['product_photos_qty'].fillna(value=final_dataframe['product_photos_qty'].mode()[0], inplace=True)
        final_dataframe['product_weight_g'].fillna(value=final_dataframe['product_weight_g'].mode()[0], inplace=True)
        final_dataframe['product_length_cm'].fillna(value=final_dataframe['product_length_cm'].mode()[0], inplace=True)
        final_dataframe['product_height_cm'].fillna(value=final_dataframe['product_height_cm'].mode()[0], inplace=True)
        final_dataframe['product_width_cm'].fillna(value=final_dataframe['product_width_cm'].mode()[0], inplace=True)
        final_dataframe['review_comment_message'].fillna(value='indisponível', inplace=True)


        final_dataframe['review_score'] = final_dataframe['review_score'].apply(lambda x: 1 if x > 3 else 0)
        final_dataframe['price_category'] = final_dataframe['price'].apply(lambda x:'expensive' if x>=139 else ('affordable' if x>=40 and x<139 else 'cheap'))
        final_dataframe = final_dataframe[final_dataframe['order_status'] != 'canceled']
        final_dataframe['purchase_delivery_diff_per_price'] = final_dataframe['purchase-delivery difference']/final_dataframe['price']
        final_dataframe.drop(['shipping_limit_date','order_purchase_timestamp','order_approved_at','order_delivered_carrier_date','order_delivered_customer_date','order_estimated_delivery_date','customer_id'], axis=1, inplace=True)

        final_dataframe['review_availability'] = final_dataframe['review_comment_message'].apply(lambda x: 1 if x != 'indisponível' else 0)

        final_dataframe.to_csv(self.config.unzip_data_dir,index=False,header=True)



    def validate_all_columns(self)-> bool:
        try:
            validation_status = None

            data = pd.read_csv(self.config.unzip_data_dir)
            
            all_cols = list(data.columns)

            all_schema = self.config.all_schema.keys()

            
            for col in all_cols:
                if col not in all_schema:
                    validation_status = False
                    with open(self.config.STATUS_FILE, 'w') as f:
                        f.write(f"Validation status: {validation_status}")
                else:
                    validation_status = True
                    with open(self.config.STATUS_FILE, 'w') as f:
                        f.write(f"Validation status: {validation_status}")

            return validation_status
        
        except Exception as e:
            raise e
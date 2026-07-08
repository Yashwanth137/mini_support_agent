import logging
import pandas as pd
from typing import Optional
from config import settings
from models import Order

logger = logging.getLogger(__name__)

class OrderTools:
    def __init__(self):
        try:
            self.df = pd.read_csv(settings.orders_csv)
            # Ensure order_id is string
            self.df['order_id'] = self.df['order_id'].astype(str)
        except Exception as e:
            logger.error("Failed to load orders.csv: %s", e)
            self.df = pd.DataFrame()

    def order_exists(self, order_id: str) -> bool:
        if self.df.empty:
            return False
        return not self.df[self.df['order_id'] == order_id].empty

    def get_order(self, order_id: str) -> Optional[Order]:
        if not self.order_exists(order_id):
            return None
        
        row = self.df[self.df['order_id'] == order_id].iloc[0]
        return Order(
            order_id=row['order_id'],
            customer_name=row['customer_name'],
            product=row['product'],
            category=row['category'],
            amount_inr=row['amount_inr'],
            order_date=str(row['order_date']),
            delivery_date=str(row['delivery_date']) if 'delivery_date' in row else None,
            status=row['status'],
            payment_method=row['payment_method'],
            pincode=str(row['pincode'])
        )

    def get_status(self, order_id: str) -> Optional[str]:
        order = self.get_order(order_id)
        return order.status if order else None

    def get_delivery_date(self, order_id: str) -> Optional[str]:
        order = self.get_order(order_id)
        return order.delivery_date if order else None

    def get_category(self, order_id: str) -> Optional[str]:
        order = self.get_order(order_id)
        return order.category if order else None

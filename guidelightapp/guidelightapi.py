# from flask import Flask
from flask_restplus import Api, Resource, fields
from tortoise import Tortoise, run_async
from tortoise_orm_marshmallow import TortoiseModelSchema

app = Flask(__name__)

# Initialize Tortoise ORM and configure the database connection
db_url = 'sqlite:////tmp/test.db'
tortoise = Tortoise(db_url=db_url)
tortoise.init_db()

# Define your Tortoise ORM models
# Example:
# class Message(tortoise.models.Model):
#     id = tortoise.fields.IntField(pk=True)
#     text = tortoise.fields.CharField(max_length=100, null=False)
# Create a TortoiseModelSchema for your Tortoise ORM models
# Example:
# class MessageSchema(TortoiseModelSchema):
#     class Meta(object):
#         model = Message
#         exclude = ('_pk', )
# Initialize the Api and define your resources
# Example:
# api = Api(app)
# message_resource = api.resource('/message')
# Create Flask-Restplus models based on TortoiseModelSchema
# Example:
# message_model = api.model('Message', message_schema.dump(message).data)
# Define your resource endpoints using the Tortoise ORM models and TortoiseModelSchema
# Example:
# @api.doc('message_resource')
# class MessageResource(Resource):
#     @api.doc('get_message')
#     def get(self, message_id):
#         message = Message.filter(pk=message_id).first.execute()
#         return message_schema.dump(message)
# Register your resources
# api.add_resource(message_resource, '/message/<int:message_id>')
if __name__ == '__main__':
    run_async(app)

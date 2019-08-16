*** Descriptions:
1. assessment_1.ipynb shows data exploration and building model
2. assessment_2.ipynb shows performance comparison between pandas and postgres
3. model.py shows the whole process from pulling data to building model
4. app.py launch a simple server for model implementation


*** Example for testing server:
curl -X POST http://127.0.0.1:5000/ -d "{'Body Style': 'PA', 'Location': 'PLATA/RAMPART', 'Ticket number': 1107780811, 'Latitude': 99999.0, 'Longitude': 99999.0, 'Violation code': '8069B', 'Color': 'BK', 'Violation Description':'NO PARKING'}"

*** Version:
    Python 3.6.2
    scikit-learn==0.20.1
    pandas==0.24.2
    numpy==1.16.4
    Flask==0.12.2

*** Caveats:
This whole thing is for a quick demo, mainly showing the detailed procedures,
other than accuracy or universal implementation.

# Company Similarity Matcher

This simple python tool outputs the companies, which are most similar to a queried company in a given dataset.

# Installation
To use the tool, install the required python packages (preferably in a virtual environment):
```
pip install -r requirements.txt
```

# Usage
Run the program from this folder with the command:

```
python app/main.py
```

This expects a `company_collection.json` file to be in this folder as well.

# Data
Entries in the JSON file should have the following format:
```
{
    "name": "Example Company",
    "founded": "2000",
    "url": "example.com",
    "headquarters": "Example Country, Example State, Example City",
    "description": "Example company does anything."
}
```

Values for fields `founded` and `headquarters` can be empty, values for other fields must be present.

# Customization
The behaviour of the tool can be customized by setting the following variables in the `main.py` file:
```
# path to JSON file with company data
FILE_PATH = "./company_collection.json"

# name of pre-trained transformer model. this can be a ...
# huggingface model: https://huggingface.co/transformers/pretrained_models.html
# SentenceTransformer model: https://docs.google.com/spreadsheets/d/14QplCdTCDwEmTqrn1LH4yrbKvdogK4oQvYO1K1aPR5M/
TRANSFORMER_MODEL = "albert-large-v2"
# TRANSFORMER_MODEL = "stsb-roberta-large" # this is a SentenceTransformer model

# if SentenceTransformer model is chosen, this must be 'True'
SENTENCE_TRANSFORMER = False

# load only N first items of dataset, loads all if set to <=0
LOAD_N = -1

# weights for the different components of the similarity function
SIMILARITY_COMPONENT_WEIGHTS = {
        "description": 0.6, "founded": 0.2, "headquarters": 0.2
}
```

# Chosen approach and possible improvements
This tool only implements a very basic approach. The idea is to leverage all of the data fields, which (in my personal opinion) promise to contain information significant for a similarity measure. This fields are: `descripion`, `founded` and `headquarters`.

* __description__:
The description of the company probably contains the most information about it, but at the same time the information is also the least structured.  
My approach is to generate embeddings of the description using the `flair` library and using a pre-trained [huggingface transformers](https://huggingface.co/transformers/pretrained_models.html) or [SentenceTransformers](https://docs.google.com/spreadsheets/d/14QplCdTCDwEmTqrn1LH4yrbKvdogK4oQvYO1K1aPR5M/edit#gid=0) model. For my testing I chose the `albert-large-v2` huggingface model, because it is an autoencoding model which can compete with the state-of-the-art. Autoencoders are (AFAIK) better suited for similarity tasks as e.g. autogressors. The similarity of the computed embeddings of the company descriptions is simply determined using cosine similarity.  
Probably one would get better embeddings for this task, if the language model would have been fine-tuned on the domain data. Also the method which is used by `flair` to generate the document embeddings might not be ideal for this task. Using a `SentenceTransformer` model might also help capturing semantics of the description (this possibility is already implemented, why I decided against it see **Known limitations**).  
More information could be leveraged from the descriptions by e.g. by extracting products or economic sectors from them. This could be done with an NER model which has been trained for this kind of entities.

* __founded__:
The idea here is that companies whose founding date is closer are more similar. Therefore the similarity of two companies is determined by the absolute difference of their founding years normalized by the greatest difference in the dataset.

* __headquarters__:
The same idea is applied as for the `founded` field: Two companies, whose headquarters are closer are more similar. To get the coordinate location of the headquarters, the `geopy` package is used.

The weight which each field has for the final similarity measure, can be changed by modifying the `SIMILARITY_COMPONENT_WEIGHTS`.

## Known limitations:
* _Description similarity heavily relying on sentence structure_:  
The embeddings of the company descriptions generated by the transformer model seem to lay heavy weight on the structure of the sentence(s) used in the descriptions. Therefore sometimes matching companies whose descriptions have a similar structure, which becomes quite obvious looking at the exemplary results listed below.

## Example results:
```
Enter URL of company: farmlogs.com
farmlogs.com        | 2012 | United States, Michigan, Ann Arbor       | AgriSight, Inc., doing business as FarmLogs, provides online farm management software solutions for farmers worldwide. It offers FarmLogs Standard, FarmLogs Advantage, FarmLogs Prescriptions, and FarmLogs Flow software tools that offer a range of solutions, such as field mapping, crop health monitoring, rainfall tracking, activity tracking, nitrogen monitoring, scouting and notes, yield maps, soil composition maps, variable rate nitrogen/seed prescriptions, input planning, inventory management, and growth stage analysis. The company was founded in 2011 and is based in Ann Arbor, Michigan.
--------------------
Most similar companies:
-----
cropzilla.com       | 2012 | United States, Ohio, Westerville         | CropZilla Software, Inc. provides farm management software. Its software allows users to master plan their fields, including applications of products and equipment use from planting to harvest; a purchase list of various inputs, including seeds, fertilizers, crop protection products, fuel, etc.; a summary of field operations with acres and hours of use for farm machinery; a summary of labor needed to complete various operations; an estimate of the time needed to complete each operation based on equipment capacity, field location and lay-out, and management approach; and cost projections of equipment and labor. The company was founded in 2013 and is based in Westerville, Ohio.
-----
roadtrippers.com    | 2011 | United States, Ohio, Cincinnati          | Roadtrippers, Inc. owns and operates a website and mobile application through which users can find, book, and purchase road trip-related travel and experiences. The company maintains a database of information regarding points-of-interest categorized into taxonomy catering to road travelers such as diners, roadside attractions, and accommodation. It also publishes travel guides on road trip routes, and provides an inline text editor for community members to write their own guides. Roadtrippers, Inc. was founded in 2011 and is based in Cincinnati, Ohio.
-----
geoamps.com         | 2010 | United States, Ohio, Powell              | geoAMPS, LLC provides land rights and infrastructure asset management software solutions. It serves land/utility/pipeline, oil and gas, alternative energy, and transportation industries in the United States and Canada. The company was founded in 2010 and is based in Powell, Ohio.
-----
sensorsuite.com     | 2012 | Canada, Ontario, Mississauga             | SensorSuite Inc. provides real-time building intelligence using wireless sensors for property managers in multi-residential, commercial, industrial, and consumer markets. It offers BoilerLink, a mobile application for monitoring and controlling boilers; FridgeLink, a mobile application for fridge monitoring and predictive maintenance; and hotel solution, a mobile application for monitoring hotel rooms. The company also offers sensors that captures, processes, analyzes, and delivers real-time building intelligence to users through Web or smartphone. SensorSuite Inc. is based in Toronto, Canada.
-----
righteye.com        | 2012 | United States, Maryland, Bethesda        | RightEye, LLC provides an eye-tracking technology that captures and scores eye movements and helps to predict, evaluate, and improve outcomes of the users. The company offers RightEye Scanning Precision Eyesight Capture Solution, a custom-developed technology that records eye movement data for various training environments; and RightEye Reading Test, a tool for educators, optometrists, and specialists to identify reading disorders in students of all ages. It serves sports; military, law enforcement, and private security personnel; and health industries. The company was incorporated in 2012 and is based in Rockville, Maryland.
-----
triaxtec.com        | 2012 | United States, Connecticut, Norwalk      | Triax Technologies, Inc. develops and delivers wearable Internet of Things (IoT) technology for construction site connectivity. The company offers spot-r, a wearable technology solution that provides real-time worksite visibility. It notifies work site safety personnel of any potential safety incidents and injuries in real-time, as well as the general location through wireless zone-based location technology. Triax Technologies, Inc. was founded in 2012 and is based in Norwalk, Connecticut.
-----
vayyar.com          | 2011 | Israel, HaMerkaz, Ness Ziona             | Vayyar Imaging Ltd. provides 3D imaging sensors for applications ranging from breast cancer screening to water leakage detection, food safety monitoring, and more. The company offers WalabotDIY, a solution that allows home renovators and DIY enthusiasts to see up to 10 centimeters into drywall, cement, and other materials to determine the location of studs, pipes, wires, and rodents' nests. It serves customers in Israel and internationally. The company was founded in 2010 and is based in Yehud, Israel.
-----
sigasec.com         | 2014 | Israel, HaDarom, Beer Sheva              | SIGASEC Ltd. develops and markets cyber security solutions for industrial control systems, and supervisory control and data acquisition systems used in critical infrastructures and industrial processes. Its device-based solution provides energy, water treatment, and industrial manufacturing markets with virtually infallible early warning in the event of anomalies caused by cyber-attacks or system malfunctions. It serves oil and gas, power and electricity, industrial manufacturing, and water and wastewater industries. The company is based in Ashkelon, Israel.
-----
bespokemetrics.com  | 2016 | Canada, Ontario, Toronto                 | Bespoke Metrics Inc. provides data management and analytical model development. The company focuses on data control, model development, and user interface to provide solutions for industries to utilize the data. It offers services, which includes data capture, data organization, data cleansing, data aggregation, and data security in data control; risk analytics, pattern recognition, grouping, data testing, third-party decision making testing, machine learning, and predictive analytics in model development; tailored data visualization, creative design, new product development through data applications, improve decision making, justify resolutions, and utilize empirical resources in user interface design. The company was founded in 2008 and is based in Toronto, Canada.
-----
identifiedtech.com  | 2013 | United States, Pennsylvania, Pittsburgh  | Identified Technologies Corporation develops self-piloting aerial mapping and land survey drones. The company offers end to end data scanning, capture, access, and data analytics solutions. Its products are used for 3-D volumetric analysis, 3-D point cloud, 2-D distance measurement, contour line map, digital surface model, orthomosaic overlay, site risk prevention and response, and securely sharing HD updates. The company serves energy, mining, manufacturing, and construction services. Identified Technologies Corporation was founded in 2013 and is based in Pittsburgh, Pennsylvania.
```





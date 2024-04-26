VENV = .venv
PYTHON = $(VENV)/bin/python3
PIP = $(VENV)/bin/pip3
STREAMLIT = $(VENV)/bin/streamlit


# Need to use python 3.9 for aws lambda
$(VENV)/bin/activate: requirements.txt
	python3 -m venv $(VENV)
	$(PIP) install -r requirements.txt



chat0: $(VENV)/bin/activate
	$(STREAMLIT) run chat0.py

chat: $(VENV)/bin/activate
	$(STREAMLIT) run chat.py

chatpdf: $(VENV)/bin/activate
	$(STREAMLIT) run chatpdf.py

chatpdfemb: $(VENV)/bin/activate
	$(STREAMLIT) run chatpdfemb.py

clean:
	rm -rf __pycache__
	rm -rf $(VENV)
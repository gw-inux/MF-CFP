# Karst Spring Response — Streamlit + CFPv2

Streamlit teaching application for simulating karst-spring response with CFPv2, CFPy and FloPy.

## Repository layout

```text
karst-spring-response/
├── streamlit_app.py
├── requirements.txt
├── smoke_test.py
├── bin/
│   └── CFPv2
├── .gitignore
└── README.md
```

`bin/CFPv2` is the native Linux x86-64 executable used by Streamlit Community Cloud. The app sets its executable permissions at runtime, so uploading the repository from Windows does not require preserving the Unix execute bit manually.

CFPy is installed from the public TU Dresden GitHub repository at a pinned commit. Do not copy a second CFPy installation into this repository unless you intentionally want to develop CFPy and the Streamlit app together.

## Streamlit Community Cloud deployment

1. Create a new GitHub repository, for example `gwp-karst-spring-response`.
2. Copy all files and folders from this package into the repository root. Keep `CFPv2` inside `bin/`.
3. Commit and push the repository to GitHub.
4. In Streamlit Community Cloud, choose **Create app** and select the new repository.
5. Use branch `main` and entrypoint `streamlit_app.py`.
6. Open **Advanced settings** and select **Python 3.11**. CFPy currently declares support for Python >=3.9 and <3.12.
7. Deploy the app. No `packages.txt` is required for the supplied CFPv2 Linux build.

## Local Linux test

Create a Python 3.11 virtual environment, install dependencies, and run the smoke test:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python smoke_test.py
streamlit run streamlit_app.py
```

## Local Windows development

The Streamlit code still accepts a Windows CFPv2 executable through the `CFP_EXECUTABLE` environment variable or a local `CFPv2.exe` fallback. The repository's bundled `bin/CFPv2` is Linux-only and is intended for Streamlit Community Cloud.

## Updating CFPy later

The CFPy dependency is pinned in `requirements.txt`:

```text
CFPy-TUD @ git+https://github.com/iGW-TU-Dresden/CFPy.git@<commit>
```

When CFPy is updated deliberately, replace `<commit>` with the tested commit SHA and redeploy. Keeping a commit pin prevents an unrelated future CFPy change from silently altering a working Streamlit deployment.

# This is modified based on gdown (https://github.com/wkentaro/gdown)
# gdown is a Python package to download public or shared Google Drive files
# It does not need Google autentication, so access can be simplified
# We do not need the functionality of download. Just need to obtain file/folder
# names along with the file IDs
import requests
import urllib
import itertools
import json
import os.path as osp
import re
import warnings
import bs4
import polars as pl


MAX_NUMBER_FILES = 50

class FileURLRetrievalError(Exception):
    pass

class _GoogleDriveFile(object):
    TYPE_FOLDER = "application/vnd.google-apps.folder"

    def __init__(self, id, name, type, children=None):
        self.id = id
        self.name = name
        self.type = type
        self.children = children if children is not None else []

    def is_folder(self):
        return self.type == self.TYPE_FOLDER

def _get_session():
    sess = requests.session()
    # We need to use different user agent for folder download c.f., file
    user_agent = (
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/98.0.4758.102 Safari/537.36')
    sess.headers.update({"User-Agent": user_agent})
    return sess

def _parse_google_drive_file(url, content):
    """Extracts information about the current page file and its children."""

    folder_soup = bs4.BeautifulSoup(content, features="html.parser")
    # finds the script tag with window['_DRIVE_ivd']
    encoded_data = None
    for script in folder_soup.select("script"):
        inner_html = script.decode_contents()
        if "_DRIVE_ivd" in inner_html:
            # first js string is _DRIVE_ivd, the second one is the encoded arr
            regex_iter = re.compile(r"'((?:[^'\\]|\\.)*)'").finditer(inner_html)
            # get the second elem in the iter
            try:
                encoded_data = (
                    next(itertools.islice(regex_iter, 1, None)).group(1))
            except StopIteration:
                raise RuntimeError("Couldn't find the folder encoded JS string")
            break

    if encoded_data is None:
        raise RuntimeError(
            "Cannot retrieve the folder information from the link. "
            "You may need to change the permission to "
            "'Anyone with the link', or have had many accesses. "
            "Check FAQ in https://github.com/wkentaro/gdown?tab=readme-ov-file#faq."
        )

    # decodes the array and evaluates it as a python array
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        decoded = encoded_data.encode("utf-8").decode("unicode_escape")
        folder_arr = json.loads(decoded)

    folder_contents = [] if folder_arr[0] is None else folder_arr[0]
    sep = " - "  # unicode dash
    splitted = folder_soup.title.contents[0].split(sep)
    if len(splitted) >= 2:
        name = sep.join(splitted[:-1])
    else:
        raise RuntimeError(
            "file/folder name cannot be extracted from: {}".format(
                folder_soup.title.contents[0]))

    gdrive_file = _GoogleDriveFile(
        id=url.split("/")[-1], name=name, type=_GoogleDriveFile.TYPE_FOLDER)
    id_name_type_iter = [
        (e[0], e[2].encode("raw_unicode_escape").decode("utf-8"), e[3])
        for e in folder_contents]
    return gdrive_file, id_name_type_iter

def _download_and_parse_google_drive_link(sess, url):
    """Get folder structure of Google Drive folder URL."""

    for _ in range(2):
        # canonicalize the language into English
        if "?" in url:
            url += "&hl=en"
        else:
            url += "?hl=en"

        res = sess.get(url, verify=True)
        # need to try with canonicalized url if the original url redirects to gdrive
        url = res.url

    gdrive_file, id_name_type_iter = _parse_google_drive_file(
        url=url, content=res.text)
    for child_id, child_name, child_type in id_name_type_iter:
        # Skip folders named 'individual'
        if child_type == _GoogleDriveFile.TYPE_FOLDER and child_name in {'Individual', 'Results'}:
            continue

        if child_type != _GoogleDriveFile.TYPE_FOLDER:
            gdrive_file.children.append(
                _GoogleDriveFile(id=child_id, name=child_name, type=child_type))
        else:
            child = _download_and_parse_google_drive_link(
                sess=sess,
                url="https://drive.google.com/drive/folders/" + child_id)
            gdrive_file.children.append(child)

    return gdrive_file

def _get_individual_signal_folder_id(sess, url):
    """
    Recursively searches for a folder named 'Predictors' and returns its ID.
    If no such folder exists, returns None.
    """
    for _ in range(2):
        # Set URL to English if not already done
        url += "&hl=en" if "?" in url else "?hl=en"
        res = sess.get(url, verify=True)
        url = res.url  # update in case of redirect

    # Parse the Google Drive file and iterate over children
    gdrive_file, id_name_type_iter = _parse_google_drive_file(url=url, content=res.text)
    for child_id, child_name, child_type in id_name_type_iter:
        # Only return ID if folder is named 'Predictors'
        if child_type == _GoogleDriveFile.TYPE_FOLDER and child_name == "Predictors":
            return child_id  # Returns only the folder ID of 'Predictors'

        # If it's another folder, recursively search in it
        if child_type == _GoogleDriveFile.TYPE_FOLDER:
            result = _get_individual_signal_folder_id(
                sess=sess,
                url="https://drive.google.com/drive/folders/" + child_id
            )
            if result:
                return result

    return None

def _get_directory_structure(gdrive_file):
    """Converts a Google Drive folder structure into a local directory list."""

    directory_structure = []
    for file in gdrive_file.children:
        file.name = file.name.replace(osp.sep, "_")
        if file.is_folder():
            directory_structure.append((None, file.name))
            for i in _get_directory_structure(file):
                directory_structure.append(i)
        elif not file.children:
            directory_structure.append((file.id, file.name))

    return directory_structure

def _get_url_from_gdrive_confirmation(contents):
    url = ""
    for line in contents.splitlines():
        m = re.search(r'href="(\/uc\?export=download[^"]+)', line)
        if m:
            url = "https://docs.google.com" + m.groups()[0]
            url = url.replace("&amp;", "&")
            break
        soup = bs4.BeautifulSoup(line, features="html.parser")
        form = soup.select_one("#download-form")
        if form is not None:
            url = form["action"].replace("&amp;", "&")
            url_components = urllib.parse.urlsplit(url)
            query_params = urllib.parse.parse_qs(url_components.query)
            for param in form.findChildren("input", attrs={"type": "hidden"}):
                query_params[param["name"]] = param["value"]
                query = urllib.parse.urlencode(query_params, doseq=True)
                url = urllib.parse.urlunsplit(url_components._replace(query=query))
            break
        m = re.search('"downloadUrl":"([^"]+)', line)
        if m:
            url = m.groups()[0]
            url = url.replace("\\u003d", "=")
            url = url.replace("\\u0026", "&")
            break
        m = re.search('<p class="uc-error-subcaption">(.*)</p>', line)
        if m:
            error = m.groups()[0]
            raise FileURLRetrievalError(error)
    if not url:
        raise FileURLRetrievalError(
            "Cannot retrieve the public link of the file. "
            "You may need to change the permission to "
            "'Anyone with the link', or have had many accesses. "
            "Check FAQ in https://github.com/wkentaro/gdown?tab=readme-ov-file#faq."
        )
    return url

def _get_name_id_map(url):
    sess = _get_session()
    gdrive_file = _download_and_parse_google_drive_link(sess, url=url)
    directory_structure = _get_directory_structure(gdrive_file)
    url_prefix = 'https://drive.google.com/uc?id='
    datasets_map = {
        'SignalDoc.csv': 'signal_doc',
        'PredictorPortsFull.csv': 'op',
        'PredictorAltPorts_Deciles.zip': 'deciles_ew',
        'PredictorAltPorts_DecilesVW.zip': 'deciles_vw',
        'PredictorAltPorts_LiqScreen_ME_gt_NYSE20pct.zip': 'ex_nyse_p20_me',
        'PredictorAltPorts_LiqScreen_NYSEonly.zip': 'nyse',
        'PredictorAltPorts_LiqScreen_Price_gt_5.zip': 'ex_price5',
        'PredictorAltPorts_Quintiles.zip': 'quintiles_ew',
        'PredictorAltPorts_QuintilesVW.zip': 'quintiles_vw',
        'signed_predictors_dl_wide.zip': 'firm_char'
    }

    df = (
        pl.DataFrame(
            directory_structure, orient='row', schema=['file_id', 'name'])
        .filter(~pl.col('name').str.contains('xlsx$|docx$|txt'))
        .with_row_index()
        .with_columns(
            pl.when(
                (pl.col('file_id').is_null()) &
                (pl.col('file_id').shift(1).is_null()) &
                (pl.col('index')!=0)
            )
            .then(pl.col('name').shift(1)+'/'+pl.col('name'))
            .when(pl.col('file_id').is_null())
            .then(pl.col('name'))
            .alias('full_name')
        )
        .with_columns(
            pl.when(
                (pl.col('name')=='SignalDoc.csv') |
                (pl.col('full_name')==pl.col('name'))
            )
            .then(pl.col('name'))
            .otherwise(pl.col('full_name').forward_fill()+'/'+pl.col('name'))
            .alias('full_name')
        )
        .filter(
            (pl.col('file_id').is_not_null()) &
            (pl.col('full_name').str.contains('zip$|csv$'))
        )
        .select('name', 'full_name', 'file_id')
        .with_columns(
            file_id=url_prefix+pl.col('file_id'),
            download_name=
            pl.col('name').replace_strict(datasets_map, default=None)
        )
        .filter(pl.col('download_name').is_not_null())
    )

    # Individual signals
    signal_folder_id = _get_individual_signal_folder_id(sess, url)
    signal_folder_url = f'https://drive.google.com/embeddedfolderview?id={signal_folder_id}'
    signal_response = requests.get(signal_folder_url)
    signal_text = str(signal_response.content)

    signal_file_name = r'<div class="flip-entry-title">(.*?).csv</div>'
    signal_file_id = r'https://drive\.google\.com/file/d/([-\w]{25,})/view\?usp=drive_web'
    signal_matches = {
        'signal': re.findall(signal_file_name, signal_text),
        'file_id': re.findall(signal_file_id, signal_text)}
    df_signal = (
        pl.DataFrame(signal_matches)
        .with_columns(file_id='https://drive.google.com/uc?id='+pl.col('file_id'))
    )
    return df, df_signal

def _get_readable_link(url):
    sess = _get_session()
    res = sess.get(url, verify=True)
    readable_link = _get_url_from_gdrive_confirmation(res.text)
    return readable_link

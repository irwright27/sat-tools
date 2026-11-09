import importlib
from io import BytesIO
from urllib.error import HTTPError, URLError
import xml.etree.ElementTree as ET

import pytest

load = importlib.import_module('s2_tools.load')
URL = 'https://sentinel2l2a01.blob.core.windows.net/sentinel2-l2/item.xml'
XML = b'<root><BOA_QUANTIFICATION_VALUE>10000</BOA_QUANTIFICATION_VALUE></root>'


@pytest.fixture(autouse=True)
def isolated_cache(monkeypatch, tmp_path):
    monkeypatch.setenv('S2_TOOLS_CACHE_DIR', str(tmp_path))
    monkeypatch.setattr(load.time, 'sleep', lambda _: None)


def test_timeout_retry_and_cache_across_signatures(monkeypatch):
    calls = []
    def fetch(url, timeout):
        calls.append(url)
        if len(calls) == 1:
            raise URLError(TimeoutError())
        return BytesIO(XML)
    monkeypatch.setattr(load, 'urlopen', fetch)
    with pytest.warns(RuntimeWarning, match='retry'):
        assert load._metadata_xml(URL + '?sig=old', 'item').tag == 'root'
    assert load._metadata_xml(URL + '?sig=new', 'item').tag == 'root'
    assert len(calls) == 2


def test_403_refreshes_unsigned_url(monkeypatch):
    calls = []
    signed = []
    def fetch(url, timeout):
        calls.append(url)
        if len(calls) == 1:
            raise HTTPError(url, 403, 'forbidden', {}, None)
        return BytesIO(XML)
    def sign(url):
        signed.append(url)
        return url + '?sig=fresh'
    monkeypatch.setattr(load, 'urlopen', fetch)
    monkeypatch.setattr(load.planetary_computer, 'sign', sign)
    with pytest.warns(RuntimeWarning):
        load._metadata_xml(URL + '?sig=expired', 'item')
    assert signed == [URL]
    assert calls[-1] == URL + '?sig=fresh'


@pytest.mark.parametrize('code,attempts', [(404, 1), (503, 4)])
def test_http_failure_is_bounded_and_redacted(monkeypatch, code, attempts):
    calls = []
    def fetch(url, timeout):
        calls.append(url)
        raise HTTPError(url, code, 'failure', {}, None)
    monkeypatch.setattr(load, 'urlopen', fetch)
    with pytest.raises(RuntimeError) as error:
        load._metadata_xml(URL + '?sig=SECRET', 'scene-id')
    assert len(calls) == attempts
    assert 'scene-id' in str(error.value)
    assert 'SECRET' not in str(error.value)


def test_invalid_xml_is_not_cached(monkeypatch, tmp_path):
    monkeypatch.setattr(load, 'urlopen', lambda *a, **k: BytesIO(b'bad XML'))
    with pytest.raises(ET.ParseError):
        load._metadata_xml(URL, 'item')
    assert not list(tmp_path.rglob('*.xml'))


def test_corrupt_cache_is_replaced(monkeypatch, tmp_path):
    monkeypatch.setattr(load, 'urlopen', lambda *a, **k: BytesIO(XML))
    load._metadata_xml(URL, 'item')
    path = next(tmp_path.rglob('*.xml'))
    path.write_bytes(b'broken')
    assert load._metadata_xml(URL, 'item').tag == 'root'
    assert path.read_bytes() == XML


def test_non_pc_403_does_not_sign(monkeypatch):
    def fetch(url, timeout):
        raise HTTPError(url, 403, 'forbidden', {}, None)
    monkeypatch.setattr(load, 'urlopen', fetch)
    monkeypatch.setattr(load.planetary_computer, 'sign', lambda _: pytest.fail('unexpected signing'))
    with pytest.raises(RuntimeError, match='HTTP 403'):
        load._metadata_xml('https://example.org/item.xml', 'item')

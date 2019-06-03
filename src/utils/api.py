import urllib
import urllib.parse
import urllib.request
import json
import time
import sys

from urllib.error import HTTPError

RATE_INTERVAL_SEC        = 3600
MAX_REQUEST_PER_INTERVAL = 900
DELAY_AFTER_EACH_REQ_SEC = 5
#DELAY_AFTER_EACH_REQ_SEC = 15
#WAIT_ON_TOO_MANY_REQ_ERROR = 600
WAIT_ON_TOO_MANY_REQ_ERROR = 10

MAX_ENTRIES_PER_PAGE = 1000

REQUEST_QUERY = 'query'
REQUEST_JSON = 'json'
REQUEST_HTML = 'html'
REQUEST_PDF = 'pdf'
# Some JSON fields
DOCUMENTS_KEY = 'documents'
DOCUMENTS_QTY_KEY = 'totalNumRecords'
FILE_FORMATS_KEY  = 'fileFormats'
COMMENT_ATTACHMENTS_KEY = 'attachments'
ATTACH_ORDER_KEY = 'attachmentOrderNumber'
COMMENT_JSON_TEXT_KEY = 'comment'
DOCUMENT_ID_KEY = 'documentId'
DOCKET_ID_KEY = 'docketId'
TITLE_KEY = 'title'
VALUE_KEY = 'value'
DOCUMENT_TYPE_KEY = 'documentType'
POSTED_DATE_KEY = 'postedDate'
CONTENT_TYPE_MARKER='contentType='

# List of doc formats that we can parse
DOCTYPE_LIST = ['msw', 'msw6', 'msw8', 'msw12']

class RateEnforcer:
  """A helper class to keep the number of requests below a given rate at all times.
     It is not very efficient for the large number of possible requests, but should
     be just fine if the maximum number of requests is a few thousands.
  """
  def __init__(self, duration, maxReqQty, minDelay = None):
    """A constructor.
  
      @param  duration      a period of time to which the maxReqQty applies
      @param  maxReqQty     a maximum number of requests per given period
      @param  minDelay      a minimum delay between requests

    """
    if (minDelay is None):
      minDelay = float(duration) / maxReqQty

    self._reqList = []
    self._duration = duration 
    self._maxReqQty = maxReqQty
    self._minDelay = minDelay

    print(f'We are going to use delay {self._minDelay}')

    assert(minDelay > 0)
    assert(maxReqQty > 0)
    assert(duration > 0)

  def waitForNextAvailable(self):
    """Wait till the next request can be processed. It returns immediately,
     if we have not reached the API limit.
    """

    currTime = time.time()
    # Wait
    if self._reqList and currTime < self._reqList[-1] + self._minDelay:
      waitTime = self._reqList[-1] + self._minDelay - currTime
      print(f'Waiting {waitTime} before submitting next request')
      time.sleep(waitTime)

    # Add a new request
    self._reqList.append(currTime)
    # Remove all stale requests
    periodStart = currTime - self._duration
    newReqList = [ tm for tm in self._reqList if tm >= periodStart ] 
    self._reqList = newReqList
    reqQty = len(self._reqList)

    if reqQty >= self._maxReqQty:
      # To figure out when we have to end waiting we take the start time
      # of the self.maxReqQty-th object counting from the end backwards
      # and add the duration of the rate period
      waitTime = max(self._reqList[-self._maxReqQty] + self._duration + 1 - time.time(), 0)
      print('Exceeded API rate, waiting %g (sec)' % waitTime)
      time.sleep(waitTime)

  def printStat(self):
    print('%d requests during the period of %g (sec)' % (len(self._reqList), self._reqList[-1] - self._reqList[0]))

def getUrlContent(url, attemptQty = 3, waitOnError = WAIT_ON_TOO_MANY_REQ_ERROR):
  """Retrieve a text response for a given URL. Wait and 
    make an additional attempt to retrieve on receiving
    too many requests error code (429).

    @param  url         a request URL
    @param  attemptQty  a maximum number of attempts
    @param  waitOnError wait this time after receiving HTTP 429 (too many requests).

    @return       a response text
  """

  for att in range(attemptQty): 
    try:
      resp = urllib.request.urlopen(url)
      return resp.read()
    except HTTPError as exObj:
      if exObj.code == 429:
        print('Received too many requests error, sleeping for %d (sec) before re-attempting' % (waitOnError))
        time.sleep(WAIT_ON_TOO_MANY_REQ_ERROR)
      else:
        print('Failed to retrieve URL %s error %s' % (url, exObj))
        sys.exit(1)

  raise Exception('Failed to retrieve %s after %d attempts' % (url, attemptQty))

def getUrlJson(url):
  """Retrieve and parse JSON produced by a given URL

    @param  url   a request URL

    @return       a parsed JSON respone
  """

  resp = getUrlContent(url)

  return json.loads(resp.decode('utf-8'))

# Learn about parameters from the dev console: https://regulationsgov.github.io/developers/console/

def buildRequestUrl(apiKey, reqType, params):
  """Construct a generic request URL using the API key, and parameters. 

    @param  apiKey     an API key
    @param  reqType    'query' (for search), 'json' (for document JSON), 'html' (for document HTML).
    @param  params     a set of additional params in the form of a tuple (param name/value) list. 

    @return a string representation of the request
  """
  if reqType == REQUEST_QUERY:
    reqTypePage = 'documents.json'
  elif reqType == REQUEST_JSON:
    reqTypePage = 'document.json'
  elif reqType == REQUEST_HTML or reqType == REQUEST_PDF:
    reqTypePage = 'download'
  else:
    raise Exception('Invalid request type %s' % reqType)
  
  return 'https://api.data.gov/regulations/v3/%s?api_key=%s&%s' % (reqTypePage, apiKey, urllib.parse.urlencode(params))

# Learn about parameters from the dev console: https://regulationsgov.github.io/developers/console/

def buildSearchRequest(apiKey, countsOnly, addParams, 
                       pageOffset = 0, entriesPerPage = MAX_ENTRIES_PER_PAGE):
  """Create a search request.

    @param      apiKey      an API key
    @param      countsOnly  if True, we request only the number of records. 
    @param      params     a set of additional params in the form of a tuple (param name/value) list. 
    @pageOffset an offset of the page start
    @entriesPerPage a maximum number per page to request (note that the API restriction is 1000)

    @return a string representation of the request
  """
  params = list(addParams) # Creates a new copy of the list
  params += [('po', pageOffset)]
  params += [('rpp',entriesPerPage)]

  if countsOnly:
    params += [('countsOnly', 1)]

  return buildRequestUrl(apiKey, REQUEST_QUERY, params)

# Learn about parameters from the dev console: https://regulationsgov.github.io/developers/console/

def buildDocRequest(apiKey, docId):
  """Create a request to retrieve a document.

    @param      apiKey      an API key.
    @param      docId       a document ID.

    @return a string representation of the request
  """

  params = [(DOCUMENT_ID_KEY, docId)] 

  return buildRequestUrl(apiKey, REQUEST_JSON, params) 
                           
# Learn about parameters from the dev console: https://regulationsgov.github.io/developers/console/

def retrieveQueryDocs(apiKey, rateEnf, queryParams):
  """Retrieve all documents for a given set of query parameters 
     as a single JSON document. This may require more than
     one call to the API.

    @param  apiKey        an API key
    @param  rateEnf       a rate enforcing object
    @param  queryParams   a set of query parameters in the form of a tuple (param name/value) list. 

    @return a number of requests made and parsed JSON with two "fields": documents and totalNumRecords
  """
  t1 = time.time()

  rateEnf.waitForNextAvailable()
  firstReq = buildSearchRequest(apiKey, False, queryParams, 
                                entriesPerPage = MAX_ENTRIES_PER_PAGE)
  firstResp = getUrlJson(firstReq)

  totalQty = firstResp[DOCUMENTS_QTY_KEY]

  batchQty = int( (totalQty + MAX_ENTRIES_PER_PAGE  - 1 ) / MAX_ENTRIES_PER_PAGE)
  print('Total # of docs to retrieve %d (# of batch %d)' % (totalQty, batchQty))
  # Retrieve additional entries
  for i in range(1, batchQty):
    rateEnf.waitForNextAvailable()
    print('Retrieving batch %d out of %d' % (i, batchQty))
    req = buildSearchRequest(apiKey, False, queryParams, 
                             pageOffset = i * MAX_ENTRIES_PER_PAGE,
                             entriesPerPage = MAX_ENTRIES_PER_PAGE)


    resp = getUrlJson(req)

    firstResp[DOCUMENTS_KEY].extend(resp[DOCUMENTS_KEY])

  t2 = time.time()
  # When zero entries, there's no documents key in the result dictionary
  qty = len(firstResp[DOCUMENTS_KEY]) if DOCUMENTS_KEY in firstResp else 0
  print('Retrieved %d entries in %g (sec)' % (qty, t2 - t1))

  return firstResp

# Learn about parameters from the dev console: https://regulationsgov.github.io/developers/console/

def retrieveRules(apiKey, rateEnf, isFinal, agencyCode = 'EPA', addQueryParams = None):
  """Retrieve final or proposed rules where commenting is no longer possible.

    @param  apiKey          an API key
    @param  rateEnf         a rate enforcing object
    @param  isFinal         True for final rules and False for proposals
    @param  agencyCode      agency code
    @param  addQueryParams  additional query parameters 
  
    @return a parsed JSON with two "fields": documents and totalNumRecords
  """
  params = list(addQueryParams if addQueryParams != None else [])

  params += [ ('a' , agencyCode) ]
  params += [ ('dct' , 'FR' if isFinal else 'PR') ]
  params += [ ('cp' , 'C') ]  # comments are closed
  params += [ ('dkt' , 'R') ] # rulemaking docs

  return retrieveQueryDocs(apiKey, rateEnf, params)


def retrieveSupportingMatterial(apiKey, rateEnf, docketId, agencyCode = 'EPA', addQueryParams = None):
  """Retrieve a list of supporting documents for a docket    @param  apiKey          an API key

    @param  apiKey          an API key
    @param  rateEnf         a rate enforcing object
    @param  docketId        docket ID
    @param  agencyCode      agency code
    @param  addQueryParams  additional query parameters

    @return a parsed JSON with two "fields": documents and totalNumRecords
  """
  params = list(addQueryParams if addQueryParams != None else [])

  params += [ ('a' , agencyCode)  ]
  params += [ ('dktid', docketId) ]
  params += [ ('dct' , 'SR')      ]

  return retrieveQueryDocs(apiKey, rateEnf, params)

# Learn about parameters from the dev console: https://regulationsgov.github.io/developers/console/

def retrieveComments(apiKey, rateEnf, agencyCode = 'EPA', addQueryParams = None):
  """Retrieve final or proposed rules where commenting is no longer possible.

    @param  apiKey          an API key
    @param  rateEnf         a rate enforcing object
    @param  isFinal         True for final rules and False for proposals
    @param  agencyCode      agency code
    @param  addQueryParams  additional query parameters 
  
    @return a parsed JSON with two "fields": 
            documents and totalNumRecords
            containing all rule-making comments
  """
  params = list(addQueryParams if addQueryParams != None else [])

  params += [ ('a' , agencyCode) ]
  params += [ ('cp' , 'C')       ]  # comments are closed
  params += [ ('dkt' , 'R')      ] # rulemaking docs
  params += [ ('docst', 'Public Comment') ] 

  return retrieveQueryDocs(apiKey, rateEnf, params)


def retrieveDoc(apiKey, docId):
  """Retrieve document by its ID.
  
    @param  apiKey      an API key
    @param  docId       a document ID   

    @return a parsed document JSON.
  """
  req = buildDocRequest(apiKey, docId)

  return getUrlJson(req)

def retrieveDocHtml(apiKey, docId, attachmentNumber = None):
  """Retrieve document HTML by its ID or URL.
  
    @param  apiKey      an API key
    @param  docId       a ID
    @param  url         an optional attachment number

    @return an unparsed document HTML.
  """
  params = [(DOCUMENT_ID_KEY, docId), ('contentType', 'html')]
  if attachmentNumber != None:
    params.append( ('attachmentNumber', str(attachmentNumber) ) )
  req = buildRequestUrl(apiKey, REQUEST_HTML, params)

  print('html', req)
  try:
    res = getUrlContent(req).decode('utf-8')
  except UnicodeDecodeError as e:
  # In some rare cases utf-8 encoding seems to be broken
    res = getUrlContent(req).decode('latin1')

  return res


def retrieveDocPdf(apiKey, docId, attachmentNumber = None):
  """Retrieve document PDF by its ID or URL.
  
    @param  apiKey      an API key
    @param  docId       a ID
    @param  url         an optional attachment number

    @return PDF's byte stream.
  """
  params = [(DOCUMENT_ID_KEY, docId), ('contentType', 'pdf')]
  if attachmentNumber != None:
    params.append( ('attachmentNumber', str(attachmentNumber) ) )
  req = buildRequestUrl(apiKey, REQUEST_PDF, params)

  print('pdf', req)

  return getUrlContent(req)


def retrieveDocBinGen(apiKey, docId, ext, attachmentNumber=None):
  """Retrieve document PDF by its ID or URL.

    @param  apiKey      an API key
    @param  ext         extension type
    @param  docId       a ID
    @param  url         an optional attachment number

    @return PDF's byte stream.
  """
  params = [(DOCUMENT_ID_KEY, docId), ('contentType', ext)]
  if attachmentNumber != None:
    params.append(('attachmentNumber', str(attachmentNumber)))
  req = buildRequestUrl(apiKey, REQUEST_PDF, params)

  print(ext, req)

  return getUrlContent(req)


def getYearFromDate(dt):
  """Extracts year from the date that is supposed to be in the format like
  this one 1994-02-14T23:59:59-05:00.

  :param dt: date string, e.g., 1994-02-14T23:59:59-05:00, empty string or None
  :return: integer year or None
  """
  if dt is None or dt == '':
    return None

  return int(dt.split('-')[0])

def getPostedYearFromJSON(docJSON):
  """
  Extracts the document posted year from a JSON (see example below).
  It works in the same way for meta-short and meta-full JSON.
  This function makes a not so nice assumption about the date format
  and doesn't check if the date format is incorrect!!!

  @param docJSON a document JSON (a dictionary entry for an element
                 in a documents array (see an example below)
  @return int year value or None

  {
    "documents": [
        {
            "agencyAcronym": "EPA",
            "allowLateComment": false,
            "attachmentCount": 0,
            "commentDueDate": "1994-02-14T23:59:59-05:00",
            "commentStartDate": "2015-03-04T00:00:00-05:00",
            "docketId": "EPA-HQ-OAR-2002-0010",
            "docketTitle": "National Emission Standards for Chromium Emissions From Hard and Decorative Chromium Electroplating and Chromium Anodizing Tanks",
            "docketType": "Rulemaking",
            "documentId": "EPA-HQ-OAR-2002-0010-0035",
            "documentStatus": "Posted",
            "documentType": "Proposed Rule",
            "frNumber": "93-30115",
            "numberOfCommentsReceived": 354,
            "openForComment": false,
            "postedDate": "2015-03-04T00:00:00-05:00",
            "rin": "Not Assigned",
            "title": "National Emission Standards for Hazardous Air Pollutants; Proposed Standards for Chromium Emissions From Hard and Decorative Chromium Electroplating and Chromium Anodizing Tanks"
        }
    ]
  }

  """
  postDate = docJSON[POSTED_DATE_KEY] if POSTED_DATE_KEY in docJSON else None

  return getYearFromDate(postDate)

def hasFormat(doc, formatCode):
  """
  Checks if a specific document format is available based on the
  full-meta JSON document entry

  @param doc   a full-meta JSON one document entry
  @param formatCode a format code: html, pdf, pdftext, ...

  @return True if the format is available
  """
  res = False
  if FILE_FORMATS_KEY in doc:
    for fmt in doc[FILE_FORMATS_KEY]:
      if fmt.endswith('=%s' % formatCode):
        res = True
  return res







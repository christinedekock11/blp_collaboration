from pyspark.sql.functions import udf, col, explode, regexp_replace
from pyspark.sql.types import ArrayType, StringType
import re
from time import time

@udf(returnType=ArrayType(StringType()))
def getTemplatesRegex(wikitext):
    """Extract list of templates from wikitext for an article via simple regex.
    Known Issues:
    * Doesn't handle nested templates (just gets the first)
    -- e.g., '{{cite web|url=http://www.allmusic.com/|ref={{harvid|AllMusic}}}}' would be just web
    """
    try:
        return list(
            set([m.split('|')[0].strip() for m in re.findall('(?<=\{\{)(.*?)(?=\}\})', wikitext, flags=re.DOTALL)]))
    except Exception:
        return None


@udf(returnType=ArrayType(StringType()))
def getTemplatesRegexRelaibility(wikitext):
    """
    Same function than getTemplatesRegex, but filtered by list of templates
    TODO: Check how to call another function (getTemplatesRegex) from here.
    """
    global templates
    try:
        all_templates = list(
            set([m.split('|')[0].strip() for m in re.findall('(?<=\{\{)(.*?)(?=\}\})', wikitext, flags=re.DOTALL)]))
        matching_templates = [template for template in all_templates if template.lower() in templates]
        if len(matching_templates) > 0:
            return matching_templates
        else:
            return None
    except Exception:
        return None
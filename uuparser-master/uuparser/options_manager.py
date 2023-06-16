import os,sys,re
import copy
import random
from collections import namedtuple

from loguru import logger

from uuparser import utils

class OptionsManager(object):

    def __init__(self,options):
        """
        input: parser options
        object to harmonise the way we deal with the parser
        """

        if options.include:
            if not options.predict and not options.datadir:
                raise Exception("You need to specify --datadir")
            elif options.shared_task and not options.testdir:
                raise Exception("You need to specify --testdir")
            if options.predict and not (options.datadir or options.testdir or
                                        options.testfile):
                raise Exception("You need to specify --testdir")

        if not options.predict:
            if not options.include and not options.trainfile:
                raise Exception("If not using the --include option, you must specify your training data with --trainfile")
        else:
            if not options.include and not options.testfile:
                raise Exception("If not using the --include option, you must specify your test data with --testfile")
            if not options.modeldir:
                options.modeldir = options.outdir # set model directory to output directory by default
            model = os.path.join(options.modeldir,options.model)
            # in monoling case we check later on language by language basis
            if options.multiling and not os.path.exists(model):
                raise Exception(f"Model not found. Path tried: {model}")

        if not options.outdir:
            raise Exception("You must specify an output directory via the --outdir option")
        elif not os.path.exists(options.outdir): # create output directory if it doesn't exist
            logger.info(f"Creating output directory {options.outdir}")
            os.mkdir(options.outdir)

        if (not options.predict and not
                                        (options.rlFlag or options.rlMostFlag or
                                         options.headFlag)):
            raise Exception("Must include either head, rl or rlmost (For example, if you specified --disable-head and --disable-rlmost, you must specify --userl)")

        if (options.rlFlag and options.rlMostFlag):
            logger.warning('Switching off rlMostFlag to allow rlFlag to take precedence')
            options.rlMostFlag = False

        if not options.multiling:
            options.tbank_emb_size = 0

        options.conllu = True #default



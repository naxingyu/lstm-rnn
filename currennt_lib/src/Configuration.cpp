/******************************************************************************
 * Copyright (c) 2013 Johannes Bergmann, Felix Weninger, Bjoern Schuller
 * Institute for Human-Machine Communication
 * Technische Universitaet Muenchen (TUM)
 * D-80290 Munich, Germany
 *
 * This file is part of CURRENNT.
 *
 * CURRENNT is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * CURRENNT is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with CURRENNT.  If not, see <http://www.gnu.org/licenses/>.
 *****************************************************************************/

#include "Configuration.hpp"
#include "rapidjson/document.h"
#include "rapidjson/filestream.h"

#include <limits>
#include <fstream>
#include <sstream>

#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <boost/random/random_device.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/replace.hpp>

namespace po = boost::program_options;

#define DEFAULT_UINT_MAX std::numeric_limits<unsigned>::max(), "inf"

Configuration *Configuration::ms_instance = NULL;


namespace internal {

std::string serializeOptions(const po::variables_map &vm) 
{
    std::string s;

    for (po::variables_map::const_iterator it = vm.begin(); it != vm.end(); ++it) {
        if (it->second.value().type() == typeid(bool))
            s += it->first + '=' + boost::lexical_cast<std::string>(boost::any_cast<bool>(it->second.value()));
        else if (it->second.value().type() == typeid(unsigned))
            s += it->first + '=' + boost::lexical_cast<std::string>(boost::any_cast<unsigned>(it->second.value()));
        else if (it->second.value().type() == typeid(float))
            s += it->first + '=' + boost::lexical_cast<std::string>(boost::any_cast<float>(it->second.value()));
        else if (it->second.value().type() == typeid(double))
            s += it->first + '=' + boost::lexical_cast<std::string>(boost::any_cast<double>(it->second.value()));
        else if (it->second.value().type() == typeid(std::string))
            s += it->first + '=' + boost::any_cast<std::string>(it->second.value());

        s += ";;;";
    }

    return s;
}

void deserializeOptions(const std::string &autosaveFile, std::stringstream *ss)
{
    // open the file
    std::ifstream ifs(autosaveFile.c_str(), std::ios::binary);
    if (!ifs.good())
        throw std::runtime_error("Cannot open file");

    // calculate the file size in bytes
    ifs.seekg(0, std::ios::end);
    size_t size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);

    // read the file into a buffer
    char *buffer = new char[size + 1];
    ifs.read(buffer, size);
    buffer[size] = '\0';

    // parse the JSON file
    rapidjson::Document jsonDoc;
    if (jsonDoc.Parse<0>(buffer).HasParseError())
        throw std::runtime_error(std::string("Parse error: ") + jsonDoc.GetParseError());

    // extract the options
    if (!jsonDoc.HasMember("configuration"))
        throw std::runtime_error("Missing string 'configuration'");

    std::string s = jsonDoc["configuration"].GetString();
    (*ss) << boost::replace_all_copy(s, ";;;", "\n");
}

} // namespace internal


Configuration::Configuration(int argc, const char *argv[])
{
    if (ms_instance)
        throw std::runtime_error("Static instance of class Configuration already created");
    else
        ms_instance = this;

    std::string optionsFile;
    std::string optimizerString;
    std::string weightsDistString;
    std::string feedForwardFormatString;

    std::string trainingFileList;
    std::string validationFileList;
    std::string testFileList;
    std::string feedForwardInputFileList;

    // create the command line options
    po::options_description commonOptions("Common options");
    commonOptions.add_options()
        ("help",                                                                              "shows this help message")
        ("options_file",       po::value(&optionsFile),                                       "reads the command line options from the file")
        ("network",            po::value(&m_networkFile)      ->default_value("network.jsn"), "sets the file containing the layout and weights of the neural network")
        ("cuda",               po::value(&m_useCuda)          ->default_value(true),          "use CUDA to accelerate the computations")
        ("list_devices",       po::value(&m_listDevices)      ->default_value(false),         "display list of CUDA devices and exit")
        ("parallel_sequences", po::value(&m_parallelSequences)->default_value(1),             "sets the number of parallel calculated sequences")
        ("random_seed",        po::value(&m_randomSeed)       ->default_value(0u),            "sets the seed for the random number generator (0 = auto)")
        ;

    po::options_description feedForwardOptions("Forward pass options");
    feedForwardOptions.add_options()
        ("ff_output_format", po::value(&feedForwardFormatString)->default_value("single_csv"), "output format for output layer activations (htk, csv or single_csv)")
        ("ff_output_file", po::value(&m_feedForwardOutputFile)->default_value("ff_output.csv"), "sets the name of the output file / directory in forward pass mode (directory for htk / csv modes)")
        ("ff_output_kind", po::value(&m_outputFeatureKind)->default_value(9), "sets the parameter kind in case of HTK output (9: user, consult HTK book for details)")
        ("feature_period", po::value(&m_featurePeriod)->default_value(10), "sets the feature period in case of HTK output (in seconds)")
        ("ff_input_file",  po::value(&feedForwardInputFileList),                                  "sets the name(s) of the input file(s) in forward pass mode")
        ("revert_std",     po::value(&m_revertStd)->default_value(true), "if regression is performed, unstandardize the output activations so that features are on the original targets' scale")
        ;

    po::options_description trainingOptions("Training options");
    trainingOptions.add_options()
        ("train",               po::value(&m_trainingMode)     ->default_value(false),                 "enables the training mode")
        ("stochastic", po::value(&m_hybridOnlineBatch)->default_value(false),                 "enables weight updates after every mini-batch of parallel calculated sequences")
        ("hybrid_online_batch", po::value(&m_hybridOnlineBatch)->default_value(false),                 "same as --stochastic (for compatibility)")
        ("shuffle_fractions",   po::value(&m_shuffleFractions) ->default_value(false),                 "shuffles mini-batches in stochastic gradient descent")
        ("shuffle_sequences",   po::value(&m_shuffleSequences) ->default_value(false),                 "shuffles sequences within and across mini-batches")
        ("max_epochs",          po::value(&m_maxEpochs)        ->default_value(DEFAULT_UINT_MAX),      "sets the maximum number of training epochs")
        ("max_epochs_no_best",  po::value(&m_maxEpochsNoBest)  ->default_value(20),                    "sets the maximum number of training epochs in which no new lowest error could be achieved")
        ("validate_every",      po::value(&m_validateEvery)    ->default_value(1),                     "sets the number of epochs until the validation error is computed")
        ("test_every",          po::value(&m_testEvery)        ->default_value(1),                     "sets the number of epochs until the test error is computed")
        ("optimizer",           po::value(&optimizerString)    ->default_value("steepest_descent"),    "sets the optimizer used for updating the weights")
        ("learning_rate",       po::value(&m_learningRate)     ->default_value((real_t)1e-5, "1e-5"),  "sets the learning rate for the steepest descent optimizer")
        ("momentum",            po::value(&m_momentum)         ->default_value((real_t)0.9,  "0.9"),   "sets the momentum for the steepest descent optimizer")
        ("weight_noise_sigma",  po::value(&m_weightNoiseSigma)  ->default_value((real_t)0), "sets the standard deviation of the weight noise added for the gradient calculation on every batch")
        ("save_network",        po::value(&m_trainedNetwork)   ->default_value("trained_network.jsn"), "sets the file name of the trained network that will be produced")
        ;

    po::options_description autosaveOptions("Autosave options");
    autosaveOptions.add_options()
        ("autosave",        po::value(&m_autosave)->default_value(false), "enables autosave after every epoch")
        ("autosave_best",        po::value(&m_autosaveBest)->default_value(false), "enables autosave on best validation error")
        ("autosave_prefix", po::value(&m_autosavePrefix),                 "prefix for autosave files; e.g. 'abc/mynet-' will lead to file names like 'mynet-epoch005.autosave' in the directory 'abc'")
        ("continue",        po::value(&m_continueFile),                   "continues training from an autosave file")
        ;

    po::options_description dataFilesOptions("Data file options");
    dataFilesOptions.add_options()
        ("train_file",        po::value(&trainingFileList),                                 "sets the *.nc file(s) containing the training sequences")
        ("val_file",          po::value(&validationFileList),                               "sets the *.nc file(s) containing the validation sequences")
        ("test_file",         po::value(&testFileList),                                     "sets the *.nc file(s) containing the test sequences")
        ("train_fraction",    po::value(&m_trainingFraction)  ->default_value((real_t)1), "sets the fraction of the training set to use")
        ("val_fraction",      po::value(&m_validationFraction)->default_value((real_t)1), "sets the fraction of the validation set to use")
        ("test_fraction",     po::value(&m_testFraction)      ->default_value((real_t)1), "sets the fraction of the test set to use")
        ("truncate_seq",      po::value(&m_truncSeqLength)    ->default_value(0),         "enables training sequence truncation to given maximum length (0 to disable)")
        ("input_noise_sigma", po::value(&m_inputNoiseSigma)   ->default_value((real_t)0), "sets the standard deviation of the input noise for training sets")
        ("input_left_context", po::value(&m_inputLeftContext) ->default_value(0), "sets the number of left context frames (first frame is duplicated as necessary)")
        ("input_right_context", po::value(&m_inputRightContext)->default_value(0), "sets the number of right context frames (last frame is duplicated as necessary)")
        ("output_time_lag",   po::value(&m_outputTimeLag)->default_value(0),              "sets the time lag in the training targets (0 = predict current frame, 1 = predict previous frame, etc.)")
        ("cache_path",        po::value(&m_cachePath)         ->default_value(""),        "sets the cache path where the .nc data is cached for random access")
        ;

    po::options_description weightsInitializationOptions("Weight initialization options");
    weightsInitializationOptions.add_options()
        ("weights_dist",         po::value(&weightsDistString)   ->default_value("uniform"),            "sets the distribution type of the initial weights (uniform or normal)")
        ("weights_uniform_min",  po::value(&m_weightsUniformMin) ->default_value((real_t)-0.1, "-0.1"), "sets the minimum value of the uniform distribution")
        ("weights_uniform_max",  po::value(&m_weightsUniformMax) ->default_value((real_t)+0.1, "0.1"),  "sets the maximum value of the uniform distribution")
        ("weights_normal_sigma", po::value(&m_weightsNormalSigma)->default_value((real_t)0.1, "0.1"),   "sets the standard deviation of the normal distribution")
        ("weights_normal_mean",  po::value(&m_weightsNormalMean) ->default_value((real_t)0.0, "0"),     "sets the mean of the normal distribution")
        ;

    po::positional_options_description positionalOptions;
    positionalOptions.add("options_file", 1);

    // parse the command line
    po::options_description visibleOptions;
    visibleOptions.add(commonOptions);
    visibleOptions.add(feedForwardOptions);
    visibleOptions.add(trainingOptions);
    visibleOptions.add(autosaveOptions);
    visibleOptions.add(dataFilesOptions);
    visibleOptions.add(weightsInitializationOptions);

    po::options_description allOptions;
    allOptions.add(visibleOptions);

    po::variables_map vm;
    try {
        po::store(po::command_line_parser(argc, argv).options(allOptions).positional(positionalOptions).run(), vm);
        if (vm.count("options_file")) {
            optionsFile = vm["options_file"].as<std::string>();
            std::ifstream file(optionsFile.c_str(), std::ifstream::in);
            if (!file.is_open())
                throw std::runtime_error(std::string("Could not open options file '") + optionsFile + "'");
            po::store(po::parse_config_file(file, allOptions), vm);
        }
        po::notify(vm);
    }
    catch (const std::exception &e) {
        if (!vm.count("help"))
            std::cout << "Error while parsing the command line and/or options file: " << e.what() << std::endl;

        std::cout << "Usage: currennt [options] [options-file]" << std::endl;
        std::cout << visibleOptions;

        exit(vm.count("help") ? 0 : 1);
    }

    if (vm.count("help")) {
        std::cout << "Usage: currennt [options] [options-file]" << std::endl;
        std::cout << visibleOptions;

        exit(0);
    }

    // load options from autosave
    if (!m_continueFile.empty()) {
        try {
            std::stringstream ss;
            internal::deserializeOptions(m_continueFile, &ss);
            vm = po::variables_map();
            po::store(po::parse_config_file(ss, allOptions), vm);
            po::notify(vm);
        }
        catch (const std::exception &e) {
            std::cout << "Error while restoring configuration from autosave file: " << e.what() << std::endl;

            exit(1);
        }
    }

    // store the options for autosave
    m_serializedOptions = internal::serializeOptions(vm);

    // split the training file options
    boost::algorithm::split(m_trainingFiles, trainingFileList, boost::algorithm::is_any_of(";,"), boost::algorithm::token_compress_on);
    if (!validationFileList.empty())
        boost::algorithm::split(m_validationFiles, validationFileList, boost::algorithm::is_any_of(";,"), boost::algorithm::token_compress_on);
    if (!testFileList.empty())
        boost::algorithm::split(m_testFiles, testFileList, boost::algorithm::is_any_of(";,"), boost::algorithm::token_compress_on);
    if (!feedForwardInputFileList.empty())
        boost::algorithm::split(m_feedForwardInputFiles, feedForwardInputFileList, boost::algorithm::is_any_of(";,"), boost::algorithm::token_compress_on);

    // check the optimizer string
    if (optimizerString == "rprop")
        m_optimizer = OPTIMIZER_RPROP;
    else if (optimizerString == "steepest_descent")
        m_optimizer = OPTIMIZER_STEEPESTDESCENT;
    else {
        std::cout << "ERROR: Invalid optimizer. Possible values: steepest_descent, rprop." << std::endl;
        exit(1);
    }

    // create a random seed
    if (!m_randomSeed)
        m_randomSeed = boost::random::random_device()();

    // check the weights distribution string
    if (weightsDistString == "normal")
        m_weightsDistribution = DISTRIBUTION_NORMAL;
    else if (weightsDistString == "uniform")
        m_weightsDistribution = DISTRIBUTION_UNIFORM;
    else {
        std::cout << "ERROR: Invalid initial weights distribution type. Possible values: normal, uniform." << std::endl;
        exit(1);
    }

    // check the feedforward format string
    if (feedForwardFormatString == "single_csv")
        m_feedForwardFormat = FORMAT_SINGLE_CSV;
    else if (feedForwardFormatString == "csv")
        m_feedForwardFormat = FORMAT_CSV;
    else if (feedForwardFormatString == "htk")
        m_feedForwardFormat = FORMAT_HTK;
    else {
        std::cout << "ERROR: Invalid feedforward format string. Possible values: single_csv, csv, htk." << std::endl;
        exit(1);
    }

    // check data sets fractions
    if (m_trainingFraction <= 0 || 1 < m_trainingFraction) {
        std::cout << "ERROR: Invalid training set fraction. Should be 0 < x <= 1" << std::endl;
        exit(1);
    }
    if (m_validationFraction <= 0 || 1 < m_validationFraction) {
        std::cout << "ERROR: Invalid validation set fraction. Should be 0 < x <= 1" << std::endl;
        exit(1);
    }
    if (m_testFraction <= 0 || 1 < m_testFraction) {
        std::cout << "ERROR: Invalid test set fraction. Should be 0 < x <= 1" << std::endl;
        exit(1);
    }

    // print information about active command line options
    if (m_trainingMode) {
        std::cout << "Started in " << (m_hybridOnlineBatch ? "hybrid online/batch" : "batch") << " training mode." << std::endl;

        if (m_shuffleFractions)
            std::cout << "Mini-batches (" << m_parallelSequences << " sequences each) will be shuffled during training." << std::endl;
        if (m_shuffleSequences)
            std::cout << "Sequences will be shuffled within and across mini-batches during training." << std::endl;
        if (m_inputNoiseSigma != (real_t)0)
            std::cout << "Using input noise with a standard deviation of " << m_inputNoiseSigma << "." << std::endl;

        std::cout << "The trained network will be written to '" << m_trainedNetwork << "'." << std::endl;
        if (boost::filesystem::exists(m_trainedNetwork))
            std::cout << "WARNING: The output file '" << m_trainedNetwork << "' already exists. It will be overwritten!" << std::endl;
    }
    else {
        std::cout << "Started in forward pass mode." << std::endl;

        std::cout << "The forward pass output will be written to '" << m_feedForwardOutputFile << "'." << std::endl;
        if (boost::filesystem::exists(m_feedForwardOutputFile))
            std::cout << "WARNING: The output file '" << m_feedForwardOutputFile << "' already exists. It will be overwritten!" << std::endl;
    }

    if (m_trainingMode && !m_validationFiles.empty())
        std::cout << "Validation error will be calculated every " << m_validateEvery << " epochs." << std::endl;
    if (m_trainingMode && !m_testFiles.empty())
        std::cout << "Test error will be calculated every " << m_testEvery << " epochs." << std::endl;

    if (m_trainingMode) {
        std::cout << "Training will be stopped";
        if (m_maxEpochs != std::numeric_limits<unsigned>::max())
            std::cout << " after " << m_maxEpochs << " epochs or";
        std::cout << " if there is no new lowest validation error within " << m_maxEpochsNoBest << " epochs." << std::endl;
    }
    
    if (m_autosave) {
        std::cout << "Autosave after EVERY EPOCH enabled." << std::endl;
    }
    if (m_autosaveBest) {
        std::cout << "Autosave on BEST VALIDATION ERROR enabled." << std::endl;
    }

    if (m_useCuda)
        std::cout << "Utilizing the GPU for computations with " << m_parallelSequences << " sequences in parallel." << std::endl;
    else
        std::cout << "WARNING: CUDA option not set. Computations will be performed on the CPU!" << std::endl;

    if (m_trainingMode) {
        if (m_weightsDistribution == DISTRIBUTION_NORMAL)
            std::cout << "Normal distribution with mean=" << m_weightsNormalMean << " and sigma=" << m_weightsNormalSigma;
        else
            std::cout << "Uniform distribution with range [" << m_weightsUniformMin << ", " << m_weightsUniformMax << "]";
        std::cout << ". Random seed: " << m_randomSeed << std::endl;
    }

    std::cout << std::endl;
}

Configuration::~Configuration()
{
}

const Configuration& Configuration::instance()
{
    return *ms_instance;
}

const std::string& Configuration::serializedOptions() const
{
    return m_serializedOptions;
}

bool Configuration::trainingMode() const
{
    return m_trainingMode;
}

bool Configuration::hybridOnlineBatch() const
{
    return m_hybridOnlineBatch;
}

bool Configuration::shuffleFractions() const
{
    return m_shuffleFractions;
}

bool Configuration::shuffleSequences() const
{
    return m_shuffleSequences;
}

bool Configuration::useCuda() const
{
    return m_useCuda;
}

bool Configuration::listDevices() const
{
    return m_listDevices;
}

bool Configuration::autosave() const
{
    return m_autosave;
}

bool Configuration::autosaveBest() const
{
    return m_autosaveBest;
}

Configuration::optimizer_type_t Configuration::optimizer() const
{
    return m_optimizer;
}

int Configuration::parallelSequences() const
{
    return (int)m_parallelSequences;
}

int Configuration::maxEpochs() const
{
    return (int)m_maxEpochs;
}

int Configuration::maxEpochsNoBest() const
{
    return (int)m_maxEpochsNoBest;
}

int Configuration::validateEvery() const
{
    return (int)m_validateEvery;
}

int Configuration::testEvery() const
{
    return (int)m_testEvery;
}

real_t Configuration::learningRate() const
{
    return m_learningRate;
}

real_t Configuration::momentum() const
{
    return m_momentum;
}

const std::string& Configuration::networkFile() const
{
    return m_networkFile;
}

const std::vector<std::string>& Configuration::trainingFiles() const
{
    return m_trainingFiles;
}

const std::string& Configuration::cachePath() const
{
    return m_cachePath;
}


const std::vector<std::string>& Configuration::validationFiles() const
{
    return m_validationFiles;
}

const std::vector<std::string>& Configuration::testFiles() const
{
    return m_testFiles;
}

unsigned Configuration::randomSeed() const
{
    return m_randomSeed;
}

Configuration::distribution_type_t Configuration::weightsDistributionType() const
{
    return m_weightsDistribution;
}

real_t Configuration::weightsDistributionUniformMin() const
{
    return m_weightsUniformMin;
}

real_t Configuration::weightsDistributionUniformMax() const
{
    return m_weightsUniformMax;
}

real_t Configuration::weightsDistributionNormalSigma() const
{
    return m_weightsNormalSigma;
}

real_t Configuration::weightsDistributionNormalMean() const
{
    return m_weightsNormalMean;
}

real_t Configuration::inputNoiseSigma() const
{
    return m_inputNoiseSigma;
}

int Configuration::inputLeftContext() const
{
    return m_inputLeftContext;
}

int Configuration::inputRightContext() const
{
    return m_inputRightContext;
}

int Configuration::outputTimeLag() const
{   
    return m_outputTimeLag;
}

real_t Configuration::weightNoiseSigma() const
{
    return m_weightNoiseSigma;
}

real_t Configuration::trainingFraction() const
{
    return m_trainingFraction;
}

real_t Configuration::validationFraction() const
{
    return m_validationFraction;
}

real_t Configuration::testFraction() const
{
    return m_testFraction;
}

const std::string& Configuration::trainedNetworkFile() const
{
    return m_trainedNetwork;
}

Configuration::feedforwardformat_type_t Configuration::feedForwardFormat() const
{
    return m_feedForwardFormat;
}

real_t Configuration::featurePeriod() const
{
    return m_featurePeriod;
}

unsigned Configuration::outputFeatureKind() const
{
    return m_outputFeatureKind;
}

unsigned Configuration::truncateSeqLength() const
{
    return m_truncSeqLength;
}

const std::vector<std::string>& Configuration::feedForwardInputFiles() const
{
    return m_feedForwardInputFiles;

}

const std::string& Configuration::feedForwardOutputFile() const
{
    return m_feedForwardOutputFile;
}

const std::string& Configuration::autosavePrefix() const
{
    return m_autosavePrefix;
}

const std::string& Configuration::continueFile() const
{
    return m_continueFile;
}

bool Configuration::revertStd() const
{
    return m_revertStd;
}

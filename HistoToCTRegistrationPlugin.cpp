#include "HistoToCTRegistrationPlugin.h"

#include "HistoToCTRegistrationAlgorithm.h"

#include <ImFusion/Core/Filesystem/Directory.h>
#include <ImFusion/Core/Resource/Resource.h>
#include <ImFusion/GUI/DefaultAlgorithmController.h>


#ifdef WIN32
extern "C" __declspec(dllexport) ImFusion::ImFusionPlugin* createPlugin()
#else
extern "C" ImFusion::ImFusionPlugin* createPlugin()
#endif
{
	return new ImFusion::HistoToCTRegistrationPlugin;
}


namespace ImFusion
{
	HistoToCTRegistrationAlgorithmFactory::HistoToCTRegistrationAlgorithmFactory()
	{
		registerAlgorithm<Histology::HistoToCTRegistrationAlgorithm>("Histology;Histology to micro CT registration (github)");
	}

	AlgorithmController* HistoToCTRegistrationControllerFactory::create(Algorithm* a) const
	{
		if (auto alg = dynamic_cast<Histology::HistoToCTRegistrationAlgorithm*>(a))
			return new DefaultAlgorithmController(alg, "", true);

		return nullptr;
	}

	HistoToCTRegistrationPlugin::HistoToCTRegistrationPlugin()
	{
		m_algFactory = new HistoToCTRegistrationAlgorithmFactory;
		m_algCtrlFactory = new HistoToCTRegistrationControllerFactory;
	}

	HistoToCTRegistrationPlugin::~HistoToCTRegistrationPlugin() = default;


	const ImFusion::AlgorithmFactory* HistoToCTRegistrationPlugin::getAlgorithmFactory() { return m_algFactory; }


	const ImFusion::AlgorithmControllerFactory* HistoToCTRegistrationPlugin::getAlgorithmControllerFactory() { return m_algCtrlFactory; }

}

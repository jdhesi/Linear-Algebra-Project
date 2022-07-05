#ifndef EXCEPTIONDEF
#define EXCEPTIONDEF

/*Exception class*/

#include <string>
class Exception
{
public:
	std::string problem, summary;
	Exception(std::string sum, std::string problem);
	void DebugPrint();
};


#endif //EXCEPTIONDEF